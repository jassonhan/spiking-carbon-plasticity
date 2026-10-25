# -*- coding: utf-8 -*-
"""T1 第五轮：FA 误差广播规则 —— 全层 vs 仅末层对照（A 无更新为基线）。"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from snn_minilm import SpikingMiniLM
from bpe_tokenizer import BPETokenizer
from stage3_plastic import FAPlasticizer
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS, gen_score


def run_fa(model, tok, x, y, steps, dev, only_last=False, eta=0.2, log_every=25):
    """FA 运营：每窗 forward -> observe_error 立即结算。"""
    # 只挂最后 N 层（only_last=True 时挂 blocks[-1]）
    pl = FAPlasticizer(model, eta=eta, verbose=False)
    if only_last:
        # 限制为只更新最后一个 block：把其它层的 eta 归零（简单做法：单独 hook 太重，
        # 直接在更新处判断 —— 这里用包装：把其它层 FA 置零即可，因为 dw ∝ outer(δ, s2)，
        # 置零方式 = 只保留 blocks[-1] 的更新，通过临时 hook 覆盖太复杂；
        # 简化：pl._FA1 只留最后层并把其它层权重更新屏蔽 —— 在 observe_error 外面做）
        pass
    pl.attach()
    curve = []
    nblk = len(model.blocks)
    for k in range(steps):
        i = k % x.shape[0]
        logits = model(x[i].unsqueeze(0).to(dev))[0]            # (T,V) 触发 hooks
        tgt = y[i].to(dev)
        lg, tg = logits[:-1], tgt[1:]
        mm = tg != -100
        ce_vec = F.cross_entropy(lg, tg, reduction="none")
        ce = float(ce_vec[mm].mean())
        T = x[i].shape[0]
        per = torch.zeros(T, device=dev)
        idx = torch.arange(1, T, device=dev)[mm]
        per[idx] = ce_vec[mm].detach()
        w = per / (per.mean() + 1e-6)
        pl.observe_error(logits.detach(), tgt, token_weight=w)
        if only_last:
            # 回滚非末层更新（把前 7 层权重恢复到 only_last 快照——简化实现：见 only_last 快照逻辑）
            for bi in range(nblk - 1):
                model.blocks[bi].fc1.weight.data.copy_(snap[bi]["fc1"])
                model.blocks[bi].fc2.weight.data.copy_(snap[bi]["fc2"])
        if k % log_every == 0 or k == steps - 1:
            tot, _ = gen_score(model, tok, PROMPTS[:2], FACTS)
            curve.append((k, ce, tot))
            print("  [FA%s] win %3d | CE %.3f | overlap %d" % ("-last" if only_last else "", k, ce, tot), flush=True)
    pl.detach()
    tot, outs = gen_score(model, tok, PROMPTS, FACTS)
    return curve, tot, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--eta", type=float, default=0.2)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    x, y = slice_windows(tok, FACTS)
    print("[T1-FA] windows=%d eta=%.2f dev=%s" % (x.shape[0], args.eta, dev))

    results = {}
    for mode in ("A", "FA-all", "FA-last"):
        if mode != "A":
            model = load_final(dev)
        else:
            model = load_final(dev)
        global snap
        snap = {i: {"fc1": model.blocks[i].fc1.weight.data.clone(),
                    "fc2": model.blocks[i].fc2.weight.data.clone()} for i in range(len(model.blocks))}
        t0 = time.time()
        if mode == "A":
            curve = []
            for k in range(0, args.steps, max(1, args.steps // 8)):
                i = k % x.shape[0]
                logits = model(x[i].unsqueeze(0).to(dev))[0]
                mm = y[i].to(dev)[1:] != -100
                ce = float(F.cross_entropy(logits[:-1][mm], y[i].to(dev)[1:][mm]))
                tot, _ = gen_score(model, tok, PROMPTS[:2], FACTS)
                curve.append((k, ce, tot))
            tot, outs = gen_score(model, tok, PROMPTS, FACTS)
        else:
            curve, tot, outs = run_fa(model, tok, x, y, args.steps, dev,
                                      only_last=(mode == "FA-last"), eta=args.eta,
                                      log_every=max(1, args.steps // 8))
        results[mode] = dict(curve=curve, overlap=tot, outs=outs, sec=round(time.time() - t0, 1))
        print("  [%s] final overlap=%d (%.1fs)" % (mode, tot, results[mode]["sec"]), flush=True)

    lines = ["=== T1-FA (steps=%d eta=%.2f dev=%s) ===" % (args.steps, args.eta, dev)]
    for m in ("A", "FA-all", "FA-last"):
        lines.append("[%s] overlap=%d | %.1fs" % (m, results[m]["overlap"], results[m]["sec"]))
        for p, txt, ov in results[m]["outs"]:
            lines.append("  %s (%d) -> %s" % (p, ov, txt[:70]))
    lines.append("--- curves ---")
    for m in ("A", "FA-all", "FA-last"):
        lines.append("[%s] " % m + " | ".join("w%d:ce%.2f,ov%d" % c for c in results[m]["curve"]))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_t1_fa_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved]", outp)


if __name__ == "__main__":
    main()
