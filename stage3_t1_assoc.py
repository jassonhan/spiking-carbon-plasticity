# -*- coding: utf-8 -*-
"""T1'（轮7）：联想记忆测试 —— Hebb-LoRA，判据 = FACTS 窗 CE 下降 + 关键词 recall。"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from stage3_lora import HebbLoRA
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS, gen_score
from bpe_tokenizer import BPETokenizer

# 关键词 recall 判据：目标实体 token 是否进入生成 top 采样
KEYWORDS = ["团子", "紫色", "小鱼干", "蝴蝶", "雪", "毛线", "猫条", "雷"]


def keyword_recall(txt, kws=KEYWORDS):
    return sum(1 for kw in kws if kw in txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eta", type=float, default=0.3)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    x, y = slice_windows(tok, FACTS)
    print("[T1-assoc] windows=%d eta=%.2f r=%d dev=%s" % (x.shape[0], args.eta, args.r, dev))

    # ---- 基线 A：无更新 ----
    model = load_final(dev)
    ce0_list = []
    for i in range(x.shape[0]):
        logits = model(x[i].unsqueeze(0).to(dev))[0]
        mm = y[i].to(dev)[1:] != -100
        ce0_list.append(float(F.cross_entropy(logits[:-1][mm], y[i].to(dev)[1:][mm])))
    ce0 = sum(ce0_list) / len(ce0_list)
    tot_a, outs_a = gen_score(model, tok, PROMPTS, FACTS)
    rec_a = sum(keyword_recall(t) for _, t, _ in outs_a)
    print("[A] mean fact-CE %.3f | keyword recall %d | overlap %d" % (ce0, rec_a, tot_a))

    # ---- Hebb-LoRA ----
    model = load_final(dev)
    pl = HebbLoRA(model, r=args.r, eta=args.eta, verbose=True)
    pl.attach()
    curve = []
    nwin = x.shape[0]
    for k in range(args.steps):
        i = k % nwin
        # forward：hook 修正后的输出（注意：CE 用旁路生效的模型——这是"运营中模型"的真实水平）
        logits = model(x[i].unsqueeze(0).to(dev))[0]
        mm = y[i].to(dev)[1:] != -100
        ce_vec = F.cross_entropy(logits[:-1][mm], y[i].to(dev)[1:][mm], reduction="none")
        ce = float(ce_vec.mean())
        T = x[i].shape[0]
        per = torch.zeros(T, device=dev)
        idx = torch.arange(1, T, device=dev)[mm]
        per[idx] = ce_vec.detach()
        w = per / (per.mean() + 1e-6)
        pl.observe_update(token_weight=w)          # Hebbian 更新（用本窗活动）
        if k % max(1, args.steps // 10) == 0 or k == args.steps - 1:
            tot, _ = gen_score(model, tok, PROMPTS[:2], FACTS)
            curve.append((k, ce, tot))
            print("  win %3d | CE %.3f | overlap %d" % (k, ce, tot), flush=True)
    # 学习后：全部窗 CE 复测 + 生成
    ce1_list = []
    for i in range(x.shape[0]):
        logits = model(x[i].unsqueeze(0).to(dev))[0]
        mm = y[i].to(dev)[1:] != -100
        ce1_list.append(float(F.cross_entropy(logits[:-1][mm], y[i].to(dev)[1:][mm])))
    ce1 = sum(ce1_list) / len(ce1_list)
    pl.detach()
    tot_f, outs_f = gen_score(model, tok, PROMPTS, FACTS)
    rec_f = sum(keyword_recall(t) for _, t, _ in outs_f)

    lines = ["=== T1-assoc Hebb-LoRA (steps=%d eta=%.2f r=%d dev=%s) ===" % (args.steps, args.eta, args.r, dev)]
    lines.append("FACTS mean CE: before %.3f -> after %.3f (delta %+.3f)" % (ce0, ce1, ce1 - ce0))
    lines.append("keyword recall: A=%d -> Hebb-LoRA=%d (of %d prompts x %d kws)" % (rec_a, rec_f, len(PROMPTS), len(KEYWORDS)))
    lines.append("overlap: A=%d -> %d" % (tot_a, tot_f))
    for p, txt, ov in outs_f:
        lines.append("  %s (%d) -> %s" % (p, ov, txt[:70]))
    lines.append("--- curve ---")
    lines.append(" | ".join("w%d:ce%.2f,ov%d" % c for c in curve))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_t1_assoc_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved]", outp)


if __name__ == "__main__":
    main()
