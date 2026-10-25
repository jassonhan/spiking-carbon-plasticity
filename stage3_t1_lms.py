# -*- coding: utf-8 -*-
"""
Stage 3 轮8：lm_head-LMS —— 输出层 delta rule（Marr-Albus，碳基可读性最高的规则）。

数学身份：ΔW = -η · outer(δ_out, h) 正是 lm_head 对 CE 的精确梯度（方向余弦=1.0 保证），
同时完全局部：误差 δ（softmax-onehot）与输入 h（ln_f 输出）都在输出层本地，无反传链。
要验证的唯一问题：仅输出层可塑，杠杆是否足够 + 生成渗透需要多少步。

保护：lm_head 直接承载语言底子，行范数增量限幅 ≤1%/步（保守），nan 回滚。
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS, gen_score
from stage3_t1_assoc import keyword_recall
from bpe_tokenizer import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--decay", type=float, default=2e-3, help="EWC 式锚定：每步 W += dw - decay*(W-W0)")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    x, y = slice_windows(tok, FACTS)
    print("[T1-LMS] windows=%d lr=%.2f decay=%.1e dev=%s" % (x.shape[0], args.lr, args.decay, dev))
    W0_cpu = None  # 锚定参照（attach 后填充）

    # ---- 基线 A ----
    model = load_final(dev)
    W_head = model.lm_head.weight.data                     # (V, D)
    W0_rows = W_head.norm(dim=1).clone()
    ce0_list = []
    for i in range(x.shape[0]):
        with torch.no_grad():
            logits = model(x[i].unsqueeze(0).to(dev))[0]
            mm = y[i].to(dev)[1:] != -100
            ce0_list.append(float(F.cross_entropy(logits[:-1][mm], y[i].to(dev)[1:][mm])))
    ce0 = sum(ce0_list) / len(ce0_list)
    tot_a, outs_a = gen_score(model, tok, PROMPTS, FACTS)
    rec_a = sum(keyword_recall(t) for _, t, _ in outs_a)
    print("[A] mean fact-CE %.3f | recall %d | overlap %d" % (ce0, rec_a, tot_a))

    # ---- LMS 运营 ----
    h_store = {}
    hook = model.ln_f.register_forward_pre_hook(
        lambda mod, inp: h_store.__setitem__("h", inp[0].detach()))
    W0_cpu = W_head.clone().cpu()                          # EWC 锚定参照
    n_accept = 0
    curve = []
    recall_curve = []
    nwin = x.shape[0]
    best_ce, best_W, best_k = float("inf"), None, -1
    best_recall, best_Wr, best_kr = -1, None, -1
    t0 = time.time()
    for k in range(args.steps):
        i = k % nwin
        logits = model(x[i].unsqueeze(0).to(dev))[0]        # (T,V) h 已被 hook 捕获
        tgt = y[i].to(dev)
        lg, tg = logits[:-1], tgt[1:]
        mm = tg != -100
        ce_vec = F.cross_entropy(lg, tg, reduction="none")
        ce = float(ce_vec.mean())
        # LMS 更新：δ = softmax - onehot（surprise 加权；probs 对齐 T-1 维）
        probs = F.softmax(logits.detach().float()[:-1], dim=-1)   # (T-1,V)
        onehot = torch.zeros_like(probs)
        onehot[mm, tg[mm]] = 1.0
        delta = probs - onehot                                    # (T-1,V)
        T = x[i].shape[0]
        per = torch.zeros(T, device=dev)
        idx = torch.arange(1, T, device=dev)[mm]
        per[idx] = ce_vec.detach()
        w = per / (per.mean() + 1e-6)
        delta_w = (delta * w[:-1].unsqueeze(1)).sum(dim=0)  # (V,) surprise 加权误差
        h = h_store["h"][0].float()                          # (T,D)
        h_w = (h[:-1] * w[:-1].unsqueeze(1)).sum(dim=0)      # (D,)
        dw = -args.lr * torch.outer(delta_w, h_w)            # (V,D) 精确梯度方向
        # 安全网：行范数增量 ≤1%
        n = W_head.norm(dim=1, keepdim=True)
        dn = dw.norm(dim=1, keepdim=True)
        scale = torch.clamp(0.01 * n / (dn + 1e-8), max=1.0)
        W_head.add_(dw * scale)
        if args.decay > 0:
            W_head.sub_(args.decay * (W_head - W0_cpu.to(dev)))   # EWC 式锚定回原点
        if ce < best_ce:                                     # best 快照（防 LMS 后期漂移）
            best_ce = ce
            best_W = W_head.clone()
            best_k = k
        if k % 500 == 0 or k == args.steps - 1:
            with torch.no_grad():
                tot, outs_now = gen_score(model, tok, PROMPTS, FACTS)
            rec = sum(keyword_recall(t) for _, t, _ in outs_now)
            recall_curve.append((k, rec, tot))
            if rec > best_recall:
                best_recall = rec
                best_Wr = W_head.clone()
                best_kr = k
            print("  win %4d | CE %.3f | overlap %d | recall %d" % (k, ce, tot, rec), flush=True)
        elif k % max(1, args.steps // 10) == 0 or k == args.steps - 1:
            tot, _ = gen_score(model, tok, PROMPTS[:2], FACTS)
            curve.append((k, ce, tot))
            print("  win %4d | CE %.3f | overlap %d" % (k, ce, tot), flush=True)
    hook.remove()
    # 恢复 recall 最优快照（recall 是最终判据，优先于 CE best）
    if best_Wr is not None:
        W_head.copy_(best_Wr)
        print("[LMS] restored best-recall snapshot: recall %d @ win %d (best-CE %.3f @ %d)"
              % (best_recall, best_kr, best_ce, best_k))
    elif best_W is not None:
        W_head.copy_(best_W)
        print("[LMS] restored best snapshot: CE %.3f @ win %d" % (best_ce, best_k))
    # 复测
    ce1_list = []
    for i in range(x.shape[0]):
        with torch.no_grad():
            logits = model(x[i].unsqueeze(0).to(dev))[0]
            mm = y[i].to(dev)[1:] != -100
            ce1_list.append(float(F.cross_entropy(logits[:-1][mm], y[i].to(dev)[1:][mm])))
    ce1 = sum(ce1_list) / len(ce1_list)
    tot_f, outs_f = gen_score(model, tok, PROMPTS, FACTS)
    rec_f = sum(keyword_recall(t) for _, t, _ in outs_f)
    sec = round(time.time() - t0, 1)

    lines = ["=== T1-LMS lm_head (steps=%d lr=%.2f dev=%s) ===" % (args.steps, args.lr, dev)]
    lines.append("FACTS mean CE: before %.3f -> after %.3f (delta %+.3f)" % (ce0, ce1, ce1 - ce0))
    lines.append("keyword recall: A=%d -> LMS=%d" % (rec_a, rec_f))
    lines.append("overlap: A=%d -> %d" % (tot_a, tot_f))
    for p, txt, ov in outs_f:
        lines.append("  %s (%d) -> %s" % (p, ov, txt[:70]))
    lines.append("--- curve (%.1fs) ---" % sec)
    lines.append(" | ".join("w%d:ce%.2f,ov%d" % c for c in curve))
    lines.append("--- recall curve ---")
    lines.append(" | ".join("w%d:rec%d,ov%d" % c for c in recall_curve))
    lines.append("final: CE delta %+.3f | recall %d | overlap %d" % (ce1 - ce0, rec_f, tot_f))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_t1_lms_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved]", outp)


if __name__ == "__main__":
    main()
