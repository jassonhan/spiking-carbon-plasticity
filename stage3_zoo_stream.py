# -*- coding: utf-8 -*-
"""Stage3 扩程曲线：16 任务流（A-P 主题 zoo），400 步/任务，判别路由头。"""
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage3_gated_stream import GatedLMS
from stage3_t1_fewshot import load_final, slice_windows
from stage3_stream8 import Router8
from stage3_zoo import TASKS16


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--steps-per-task", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--order-seed", type=int, default=None)
    ap.add_argument("--ckpt", default=None, help="替代 snn34_final 的模型 ckpt")
    args = ap.parse_args()
    if args.smoke:
        args.steps_per_task = 20
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = __import__("bpe_tokenizer", fromlist=["BPETokenizer"]).BPETokenizer.load(
        os.path.join(W, "bpe_chinese_wanjuan.json"))
    data = {}
    for name, facts, prompts, kws in TASKS16:
        data[name] = slice_windows(tok, facts)
        print("[data] %s: %d windows" % (name, data[name][0].shape[0]), flush=True)
    model = load_final(dev)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
        sd = {k: v.to(dev) for k, v in ck["model"].items() if not k.endswith(".v")}
        model.load_state_dict(sd, strict=False)
        model.eval()
        print("[ckpt] loaded %s" % args.ckpt, flush=True)
    gl = GatedLMS(model, dev)
    print("[router] 训练 16 任务路由头...", flush=True)
    r16 = Router8(model, dev, tok, TASKS16)
    r16.load_into(gl)
    rows = []
    t0 = time.time()
    order = list(TASKS16)
    if args.order_seed is not None:
        import random as _rd
        _rd.Random(args.order_seed).shuffle(order)
        print("[order] %s" % "→".join(t[0] for t in order), flush=True)
    for name, facts, prompts, kws in order:
        x, y = data[name]
        gl.ensure_task(name)
        for k in range(args.steps_per_task):
            i = k % x.shape[0]
            gl.learn_step(name, x[i], y[i], args.lr)
        gl.set_key(name, x)
        for t in gl.deltas:
            xt, yt = data[t]
            ce, gacc = gl.pool_ce(xt, yt, true_task=t)
            rows.append(("%s@%s" % (t, name), ce, gacc))
        seen = [r for r in rows if r[0].split("@")[0] == r[0].split("@")[1]]
        print("  [%s 学完] 已见任务 acc: %s" % (name, " ".join(
            "%s %.2f" % (r[0].split("@")[0], r[2]) for r in seen)), flush=True)

    learn_ce, final_ce = {}, {}
    for name, ce, g in rows:
        t, ph = name.split("@")
        if t == ph:
            learn_ce[t] = ce
        final_ce[t] = ce
    lines = ["=== Gated-LMS 16-task stream (zoo, lr=%.2f) ===" % args.lr]
    drifts = {}
    for t in learn_ce:
        d = final_ce[t] - learn_ce[t]
        drifts[t] = d
        lines.append("  %s: 学到 %.3f -> 流末 %.3f (漂移 %+.3f)" %
                     (t, learn_ce[t], final_ce[t], d))
    mean_d = sum(drifts.values()) / len(drifts)
    zero = sum(1 for d in drifts.values() if abs(d) < 0.01)
    lines.append("平均漂移: %+.3f | 精确零漂移任务 %d/16" % (mean_d, zero))
    lines.append("扩程曲线: 4任务 +0.004 → 8任务 +0.046 → 16任务 %+.3f" % mean_d)
    lines.append("(%.0fs)" % (time.time() - t0))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_stream16_report%s.txt" %
                        ("" if args.order_seed is None else "_s%d" % args.order_seed))
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
