# -*- coding: utf-8 -*-
"""Stage3 T1 —— 英语新领域适应（真实 60M MLP 模型）。

设计依据（2026-11-20 结论）：同分布 held-out 上无监督 Hebbian 无收益；
正确场景 = 模型没学过的领域。模型纯中文训练（latin≤0.01 过滤），英文是完美新领域。

测试项：
  1) 英文 base CE（预期很高）
  2) 调质门控 ΔW 少样本适应（η 扫描 1e-10 ~ 1e-7）
  3) 中文保留度（适应后 clean val 漂移 —— ΔW 的核心卖点：不破坏长期权重）
  4) 全参微调对照（同 token 预算 BPTT）
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from bpe_tokenizer import BPETokenizer                     # noqa: E402
from mlp_stage3 import MlpVariant, convert_to_plastic      # noqa: E402
from stage3_modulator import PlasticLinear, PlasticityConfig  # noqa: E402

DEV = "cuda"
WIN = 128
BEST = os.path.join(W, "SNN34_distill", "snn34_mlp.best.pt")
EN_NPY = r"E:\english_corpus\wiki_en_bpe.npy"
ZH_VAL = r"C:\wanjuan_local\clean_val_bpe.npy"

tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
VOCAB = tok.vocab_size

en = np.load(EN_NPY, mmap_mode="r")
zh = np.load(ZH_VAL, mmap_mode="r")
rng = np.random.default_rng(2026)
# 适应集 8 条（前 40% 区间），测试集 32 条（后 60% 区间，不重叠）
lo, mid = 0, int(len(en) * 0.4)
adapt = [np.asarray(en[s:s + WIN], dtype=np.int64)
         for s in rng.integers(lo, mid, size=8)]
en_test = np.stack([np.asarray(en[s:s + WIN], dtype=np.int64)
                    for s in rng.integers(mid, len(en) - WIN, size=32)])
zh_test = np.stack([np.asarray(zh[i * WIN:(i + 1) * WIN], dtype=np.int64)
                    for i in range(32)])

model = MlpVariant(VOCAB, 640, 10, 8, WIN, alpha=1.0).to(DEV)
ck = torch.load(BEST, map_location=DEV, weights_only=False)
model.load_state_dict({k: v for k, v in ck["model"].items() if not k.endswith(".v")},
                      strict=False)
model = convert_to_plastic(model, PlasticityConfig(clamp_dw=0.002))
plastics = [m for b in model.blocks for m in b.modules() if isinstance(m, PlasticLinear)]
print("[model] %s | PlasticLinear=%d" % (os.path.basename(BEST), len(plastics)))


@torch.no_grad()
def ce(x, bs=8):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        b = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(DEV)
        lg = model(b)
        lo_ = F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB), b[:, 1:].reshape(-1))
        tot += lo_.item() * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / n


def set_plastic(on):
    for b in model.blocks:
        b.plastic_on = on


en_base = ce(en_test)
zh_base = ce(zh_test)
print("\n[base] 英文 CE = %.4f | 中文 val CE = %.4f" % (en_base, zh_base))
print("  （英文 CE 比中文高 %.2f —— 新领域陌生度）" % (en_base - zh_base))


def trial(eta):
    for m in plastics:
        m.cfg.eta = eta
        m.reset_plastic()
    set_plastic(False)
    with torch.no_grad():
        for s in adapt:
            xb = torch.from_numpy(s[None, :]).to(DEV)
            lg = model(xb)
            sup = F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB),
                                  xb[:, 1:].reshape(-1)).item()
            set_plastic(True)
            for b in model.blocks:
                b.m_const = b.novelty_m(sup, b.cfg)
            model(xb)
            set_plastic(False)
    dw = max(m.dw.abs().max().item() for m in plastics)
    return ce(en_test), ce(zh_test), dw


print("\n[T1-English] η 扫描（8-shot 英文适应，零 BPTT）:")
rows = []
for eta in (1e-10, 5e-10, 1e-9, 5e-9, 5e-8, 1e-7):
    ce_en, ce_zh, dw = trial(eta)
    rows.append((eta, ce_en, ce_zh, dw))
    print("  η=%.0e: 英文 CE=%.4f (%+.4f) | 中文 CE=%.4f (%+.4f) | |dw|max=%.5f %s"
          % (eta, ce_en, ce_en - en_base, ce_zh, ce_zh - zh_base, dw,
             "OK" if ce_en < en_base else "X"))

# 全参微调对照（同 8 序列 × 1 遍 BPTT，lr 1e-5 —— 保守）
set_plastic(False)
for m in plastics:
    m.reset_plastic()
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
for step in range(8):
    s = adapt[step % len(adapt)]
    xb = torch.from_numpy(s[None, :]).to(DEV)
    lg = model(xb)
    loss = F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB), xb[:, 1:].reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
ft_en, ft_zh = ce(en_test), ce(zh_test)

best_row = min(rows, key=lambda r: r[1])
print("\n=== 英语 T1 结果汇总 ===")
print("  英文 base            : %.4f" % en_base)
print("  ΔW 最优 (η=%.0e)     : %.4f (%+.4f) | 中文漂移 %+.4f"
      % (best_row[0], best_row[1], best_row[1] - en_base, best_row[2] - zh_base))
print("  全参微调 (8 步 BPTT) : %.4f (%+.4f) | 中文漂移 %+.4f"
      % (ft_en, ft_en - en_base, ft_zh - zh_base))
