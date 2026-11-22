# -*- coding: utf-8 -*-
"""Stage3 完整闭环验证：冻结底座 → 连续学 4 任务 → 全任务回访。

任务：A=英文维基 / B=中文新闻 / C=中文法律 / D=中文通用(广告/公告段)
协议：
  阶段0 base（测 4 任务）
  阶段1-4 顺序学习 A→B→C→D（每任务 10 条 × 1 遍，FA lr=3e-3 decay=0.02）
  阶段5-8 依次回访重学 A→B→C→D（测 savings 与最终保留）
指标：
  - 每阶段 4 任务的 CE
  - 总遗忘 = 阶段4 相对 base 的 4 任务平均退化（学习期结束时的遗忘峰值）
  - 最终保留 = 阶段8 相对 base 的 4 任务平均变化
  - savings = 第二次学同一任务的改善 / 第一次（>1 表示学得更快）
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
zh = np.memmap(r"C:\wanjuan_local\wanjuan_extra_bpe.bin", dtype=np.int32, mode="r")
seg = len(zh) // 16
rng = np.random.default_rng(2026)

TASKS = {}
# A 英文
TASKS["A_en"] = dict(
    train=[np.asarray(en[s:s + WIN], dtype=np.int64)
           for s in rng.integers(0, int(len(en) * 0.3), size=10)],
    test=np.stack([np.asarray(en[s:s + WIN], dtype=np.int64)
                   for s in rng.integers(int(len(en) * 0.4), len(en) - WIN, size=16)]))
# B 新闻 seg2 / C 法律 seg3 / D 通用 seg10
for name, si in (("B_news", 2), ("C_law", 3), ("D_gen", 10)):
    lo, hi = si * seg, (si + 1) * seg
    TASKS[name] = dict(
        train=[np.asarray(zh[s:s + WIN], dtype=np.int64)
               for s in rng.integers(lo, lo + seg // 3, size=10)],
        test=np.stack([np.asarray(zh[s:s + WIN], dtype=np.int64)
                       for s in rng.integers(lo + seg // 3, hi, size=16)]))

ORDER = ["A_en", "B_news", "C_law", "D_gen"]


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
    row = {k: ce(m, TASKS[k]["test"]) for k in ORDER}
    print("%-16s " % tag + "  ".join("%s=%.4f" % (k.split("_")[1], v) for k, v in row.items()),
          flush=True)
    return row


m = load_fa_from(BEST, VOCAB, 640, 10, 8, WIN, DEV)
m.reset_dw()
LR, DECAY, CLAMP = 3e-3, 0.02, 0.02

print("=== Stage3 闭环：冻结底座 + FA(lr=%.0e, decay=%.2f) 连续学 4 任务 → 全回访 ===" % (LR, DECAY))
rows = {}
rows["base"] = snap(m, "0. base")
first_gain = {}
for t in ORDER:
    for s in TASKS[t]["train"]:
        m.fa_adapt(torch.from_numpy(s[None, :]).to(DEV), lr=LR, clamp=CLAMP, decay=DECAY)
    prev = rows[list(rows)[-1]][t]
    rows["学" + t.split("_")[1]] = snap(m, "%d. 学%s" % (len(rows), t.split("_")[1]))
    first_gain[t] = prev - rows[list(rows)[-1]][t]
    # 注：prev 用上一阶段该任务的值（学习前基线）

peak = rows[list(rows)[-1]]
print("\n--- 学习期结束（阶段4）遗忘峰值 ---")
for t in ORDER:
    print("  %s: %+.4f" % (t, peak[t] - rows["base"][t]))

second_gain = {}
for t in ORDER:
    before = ce(m, TASKS[t]["test"])
    for s in TASKS[t]["train"]:
        m.fa_adapt(torch.from_numpy(s[None, :]).to(DEV), lr=LR, clamp=CLAMP, decay=DECAY)
    row = snap(m, "%d. 回访%s" % (len(rows), t.split("_")[1]))
    rows["回" + t.split("_")[1]] = row
    second_gain[t] = before - row[t]

final = rows[list(rows)[-1]]
print("\n=== 汇总 ===")
print("任务     base     峰值遗忘   最终变化   savings(二/一)")
for t in ORDER:
    g1 = first_gain[t]
    g2 = second_gain[t]
    sv = (g2 / g1) if abs(g1) > 1e-6 else float("nan")
    print("%-8s %.4f   %+.4f    %+.4f    %s"
          % (t, rows["base"][t], peak[t] - rows["base"][t],
             final[t] - rows["base"][t], ("%.2fx" % sv) if sv == sv else "n/a"))
avg_peak = np.mean([peak[t] - rows["base"][t] for t in ORDER])
avg_final = np.mean([final[t] - rows["base"][t] for t in ORDER])
print("\n平均: 峰值遗忘 %+.4f | 最终变化 %+.4f" % (avg_peak, avg_final))
