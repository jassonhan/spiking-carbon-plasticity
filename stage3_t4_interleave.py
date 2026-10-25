# -*- coding: utf-8 -*-
"""
Stage 3 T4: 干扰稳定性——多任务并行（A+B 交错）vs 顺序学习（T2 v2b）对照。
问题：从零并行学两个任务，LMS 的输出层共享权重会不会互相拉扯？
对照基准：T2 v2b 顺序学习（A 先 B 后+重放）：A 7.18 / B 6.53。
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage3_t2_sequence import lms_phase, eval_task, FACTS_B, PROMPTS_B, KEYWORDS_B
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS as PROMPTS_A
from stage3_t1_assoc import KEYWORDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1600, help="并行学习总步数（=T2 两阶段之和）")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--decay", type=float, default=2e-3)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 60
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer = __import__("bpe_tokenizer", fromlist=["BPETokenizer"]).BPETokenizer
    tok = tok.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    xA, yA = slice_windows(tok, FACTS)
    xB, yB = slice_windows(tok, FACTS_B)
    model = load_final(dev)

    # 基线
    ceA0, recA0, ovA0, _ = eval_task(model, tok, xA, yA, PROMPTS_A, FACTS, KEYWORDS)
    ceB0, recB0, ovB0, _ = eval_task(model, tok, xB, yB, PROMPTS_B, FACTS_B, KEYWORDS_B)
    print("[T4] baseline: A %.3f / B %.3f" % (ceA0, ceB0))

    # 并行学习：A+B 窗交错（每窗轮换）
    xI = torch.cat([xA, xB], dim=0)
    yI = torch.cat([yA, yB], dim=0)
    t0 = time.time()
    lms_phase(model, tok, xI, yI, args.steps, args.lr, args.decay, "interleave")
    ceA1, recA1, ovA1, _ = eval_task(model, tok, xA, yA, PROMPTS_A, FACTS, KEYWORDS)
    ceB1, recB1, ovB1, _ = eval_task(model, tok, xB, yB, PROMPTS_B, FACTS_B, KEYWORDS_B)
    print("  并行后: A %.3f (rec %d) | B %.3f (rec %d) | %.0fs"
          % (ceA1, recA1, ceB1, recB1, time.time() - t0))

    lines = []
    lines.append("=== T4 interference: interleaved A+B (LMS lr=%.2f decay=%.1e %d steps) ==="
                 % (args.lr, args.decay, args.steps))
    lines.append("%-18s %8s %6s %6s" % ("metric", "CE", "recall", "overlap"))
    for name, ce, rec, ov in [("baseline-A", ceA0, recA0, ovA0), ("baseline-B", ceB0, recB0, ovB0),
                              ("interleaved-A", ceA1, recA1, ovA1), ("interleaved-B", ceB1, recB1, ovB1)]:
        lines.append("%-18s %8.3f %6d %6d" % (name, ce, rec, ov))
    # 对照：T2 v2b 顺序+重放（A 7.177 / B 6.532）；无重放顺序（A 9.668 / B 6.401）
    lines.append("")
    lines.append("对照 T2 顺序学习: 无重放 A 9.668/B 6.401 | 重放20%% A 7.177/B 6.532")
    lines.append("并行 vs 顺序+重放: A %+.3f / B %+.3f" % (ceA1 - 7.177, ceB1 - 6.532))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_t4_interleave_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
