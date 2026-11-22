# -*- coding: utf-8 -*-
"""Stage3 T3：数据效率曲线 —— FA vs 全参微调（同等 token 预算）。

设计：
  - 适应预算 K ∈ {1,2,5,20,80} 条序列（每序列 128 token）
  - FA: 每序列 1 遍 fa_adapt（局部，只动 dw）
  - 微调: 每序列 1 步 BPTT（lr 1e-5，只动 base 权重）
  - 指标：英文 held-out CE（主线指标） + 中文漂移（防遗忘）

T3 及格线（原设计）：新任务上用 1/10 token 达到全参微调同等适应度。
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from bpe_tokenizer import BPETokenizer                    # noqa: E402
from fa_stage3 import FAMlpVariant, load_fa_from          # noqa: E402

DEV = "cuda"
WIN = 128
BEST = os.path.join(W, "SNN34_distill", "snn34_mlp.best.pt")
tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
VOCAB = tok.vocab_size

en = np.load(r"E:\english_corpus\wiki_en_bpe.npy", mmap_mode="r")
zh = np.load(r"C:\wanjuan_local\clean_val_bpe.npy", mmap_mode="r")
rng = np.random.default_rng(2026)
pool = [np.asarray(en[s:s + WIN], dtype=np.int64)
        for s in rng.integers(0, int(len(en) * 0.4), size=80)]
en_test = np.stack([np.asarray(en[s:s + WIN], dtype=np.int64)
                    for s in rng.integers(int(len(en) * 0.4), len(en) - WIN, size=16)])
zh_test = np.stack([np.asarray(zh[i * WIN:(i + 1) * WIN], dtype=np.int64)
                    for i in range(16)])

Bs = [1, 2, 5, 20, 80]
FA_LR = 3e-3
FT_LR = 1e-5


@torch.no_grad()
def ce(m, x, bs=4):
    m.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        b = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(DEV)
        lg = m(b)
        lo = F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB), b[:, 1:].reshape(-1))
        tot += lo.item() * b.shape[0]
        n += b.shape[0]
    m.train()
    return tot / n


print("加载 base 模型...", flush=True)
m0 = load_fa_from(BEST, VOCAB, 640, 10, 8, WIN, DEV)
en_base, zh_base = ce(m0, en_test), ce(m0, zh_test)
print("base: EN=%.4f ZH=%.4f\n" % (en_base, zh_base), flush=True)
del m0
torch.cuda.empty_cache()

fa_curve, ft_curve = [], []
for K in Bs:
    # ---- FA ----
    m = load_fa_from(BEST, VOCAB, 640, 10, 8, WIN, DEV)
    m.reset_dw()
    for i in range(K):
        m.fa_adapt(torch.from_numpy(pool[i][None, :]).to(DEV), lr=FA_LR, clamp=0.02)
    e, z = ce(m, en_test), ce(m, zh_test)
    fa_curve.append((K, e, z))
    print("FA  K=%3d (%6d tok): EN=%.4f (%+.4f) ZH=%+.4f"
          % (K, K * WIN, e, e - en_base, z - zh_base), flush=True)
    del m
    torch.cuda.empty_cache()

    # ---- 微调（同 K：每序列 1 步 BPTT） ----
    m = load_fa_from(BEST, VOCAB, 640, 10, 8, WIN, DEV)
    m.reset_dw()
    opt = torch.optim.AdamW(m.parameters(), lr=FT_LR)
    for i in range(K):
        xb = torch.from_numpy(pool[i][None, :]).to(DEV)
        lg = m(xb[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, VOCAB), xb[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    e, z = ce(m, en_test), ce(m, zh_test)
    ft_curve.append((K, e, z))
    print("FT  K=%3d (%6d tok): EN=%.4f (%+.4f) ZH=%+.4f"
          % (K, K * WIN, e, e - en_base, z - zh_base), flush=True)
    del m, opt
    torch.cuda.empty_cache()

print("\n=== T3 数据效率（英文 CE 改善 / 中文漂移）===")
print("  K    tok     FA                FT")
for (K, fe, fz), (_, te, tz) in zip(fa_curve, ft_curve):
    print("  %-4d %-7d %+.4f/%+.4f   %+.4f/%+.4f"
          % (K, K * WIN, fe - en_base, fz - zh_base, te - en_base, tz - zh_base))

# 数据效率比：FA 达到 FT 最大改善 90% 所需 token
fa_best = min(r[1] for r in fa_curve)
ft_best = min(r[1] for r in ft_curve)
print("\n  FA 最佳改善 = %+.4f @ K=%d | FT 最佳 = %+.4f @ K=%d"
      % (fa_best - en_base, [r[0] for r in fa_curve if r[1] == fa_best][0],
         ft_best - en_base, [r[0] for r in ft_curve if r[1] == ft_best][0]))
