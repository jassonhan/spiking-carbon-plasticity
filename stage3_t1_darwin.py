# -*- coding: utf-8 -*-
"""T1 第六轮：Darwin-LoRA —— 随机试探 + 奖励选择（神经达尔文主义）。"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from stage3_lora import DarwinLoRA
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS, gen_score
from bpe_tokenizer import BPETokenizer


def run_darwin(model, tok, x, y, steps, dev, r=8, sigma=0.05, log_every=25):
    dl = DarwinLoRA(model, r=r, sigma=sigma, verbose=True)
    dl.attach()
    curve = []
    for k in range(steps):
        i = k % x.shape[0]
        dl.step_begin()                                   # 快照 + 提案 θ'
        logits = model(x[i].unsqueeze(0).to(dev))[0]      # 评估 θ'
        lg, tg = logits[:-1], y[i].to(dev)[1:]
        mm = tg != -100
        ce = float(F.cross_entropy(lg[mm], tg[mm]))
        accepted, base = dl.evaluate(ce)                  # 选择
        if k % log_every == 0 or k == steps - 1:
            tot, _ = gen_score(model, tok, PROMPTS[:2], FACTS)
            curve.append((k, ce, tot))
            print("  [DL] win %3d | CE %.3f | %s | overlap %d | %s"
                  % (k, ce, "ACC" if accepted else "rej", tot, dl.summary(10)), flush=True)
    dl.detach()
    tot, outs = gen_score(model, tok, PROMPTS, FACTS)
    return curve, tot, outs, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    x, y = slice_windows(tok, FACTS)
    print("[T1-Darwin] windows=%d r=%d sigma=%.3f dev=%s" % (x.shape[0], args.r, args.sigma, dev))

    # A 基线
    model = load_final(dev)
    base_curve = []
    for k in range(0, args.steps, max(1, args.steps // 8)):
        i = k % x.shape[0]
        logits = model(x[i].unsqueeze(0).to(dev))[0]
        mm = y[i].to(dev)[1:] != -100
        ce = float(F.cross_entropy(logits[:-1][mm], y[i].to(dev)[1:][mm]))
        base_curve.append((k, ce))
    tot_a, outs_a = gen_score(model, tok, PROMPTS, FACTS)

    # Darwin-LoRA
    model = load_final(dev)
    t0 = time.time()
    curve, tot, outs, dl = run_darwin(model, tok, x, y, args.steps, dev,
                                      r=args.r, sigma=args.sigma,
                                      log_every=max(1, args.steps // 10))
    sec = round(time.time() - t0, 1)

    lines = ["=== T1-DarwinLoRA (steps=%d r=%d sigma=%.3f dev=%s) ===" % (args.steps, args.r, args.sigma, dev)]
    lines.append("[A] overlap=%d | CE curve: %s" % (tot_a, " ".join("%.2f" % c for _, c in base_curve)))
    lines.append("[DL] overlap=%d | %.1fs | accept rate %.0f%% (%d/%d) | final %s"
                 % (tot, sec, 100 * dl.n_accept / max(1, dl.n_total), dl.n_accept, dl.n_total, dl.summary()))
    for p, txt, ov in outs:
        lines.append("  %s (%d) -> %s" % (p, ov, txt[:70]))
    lines.append("--- DL CE curve ---")
    lines.append(" | ".join("w%d:ce%.2f,ov%d" % c for c in curve))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_t1_darwin_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved]", outp)


if __name__ == "__main__":
    main()
