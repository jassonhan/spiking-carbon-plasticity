# -*- coding: utf-8 -*-
"""Stage3 T2：序列记忆 A→B→A（快权重动力学）。

A = 英文维基，B = 中文 news 段（两个不同领域）
协议：base → 学A → 学B → 回A → 学B → 回A（每阶段 10 条 × 1 遍，FA + decay）
观测：
  - 各阶段 A / B 的 CE（看干扰与恢复）
  - 重学效应（savings）：第二次学 A 的改善幅度 vs 第一次
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from bpe_tokenizer import BPETokenizer                    # noqa: E402
from fa_stage3 import load_fa_from                        # noqa: E402

DEV = "cuda"
WIN = 128
BEST = os.path.join(W, "SNN34_distill", "snn34_mlp.best.pt")
tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
VOCAB = tok.vocab_size

en = np.load(r"E:\english_corpus\wiki_en_bpe.npy", mmap_mode="r")
zhbin = np.memmap(r"C:\wanjuan_local\wanjuan_extra_bpe.bin", dtype=np.int32, mode="r")
seg = len(zhbin) // 16
rng = np.random.default_rng(2026)

# A: 英文（前 40% 内取训练/测试）
A_train = [np.asarray(en[s:s + WIN], dtype=np.int64)
           for s in rng.integers(0, int(len(en) * 0.3), size=10)]
A_test = np.stack([np.asarray(en[s:s + WIN], dtype=np.int64)
                   for s in rng.integers(int(len(en) * 0.4), len(en) - WIN, size=16)])
# B: 中文 news 段（seg2）
B_train = [np.asarray(zhbin[s:s + WIN], dtype=np.int64)
           for s in rng.integers(2 * seg, 2 * seg + seg // 3, size=10)]
B_test = np.stack([np.asarray(zhbin[s:s + WIN], dtype=np.int64)
                   for s in rng.integers(2 * seg + seg // 3, 3 * seg, size=16)])
# C: 中文法律段（seg3-4，干扰任务二）
C_train = [np.asarray(zhbin[s:s + WIN], dtype=np.int64)
           for s in rng.integers(3 * seg, 3 * seg + seg // 3, size=10)]
C_test = np.stack([np.asarray(zhbin[s:s + WIN], dtype=np.int64)
                   for s in rng.integers(3 * seg + seg // 3, 4 * seg, size=16)])


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


def snap(m, tag):
    a, b, c = ce(m, A_test), ce(m, B_test), ce(m, C_test)
    print("%-18s A(en)=%.4f  B(news)=%.4f  C(law)=%.4f" % (tag, a, b, c), flush=True)
    return a, b, c


m = load_fa_from(BEST, VOCAB, 640, 10, 8, WIN, DEV)
m.reset_dw()
LR, DECAY, CLAMP = 3e-3, 0.02, 0.02

print("=== T2 序列记忆 A→B→A→C→A（FA lr=%.0e decay=%.2f，每阶段 10 条×1 遍）===" % (LR, DECAY))
base = snap(m, "base")
hist = [("base", base)]

def learn(seqs, tag):
    for s in seqs:
        m.fa_adapt(torch.from_numpy(s[None, :]).to(DEV), lr=LR, clamp=CLAMP, decay=DECAY)
    return snap(m, tag)

learn(A_train, "1. 学A(英)")
learn(B_train, "2. 学B(新闻)")
learn(C_train, "3. 学C(法律)")
learn(A_train, "4. 回到A(英)")
learn(B_train, "5. 回到B(新闻)")
learn(A_train, "6. 再回A(英)")

print("\n=== 判读 ===")
print("  A 的 CE 轨迹: ", " -> ".join("%.3f" % h[1][0] for h in hist) if False else "")
