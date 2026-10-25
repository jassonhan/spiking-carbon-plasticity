# -*- coding: utf-8 -*-
"""对照实验 C：BPTT 训练同样的 LoRA 旁路 —— 分离「杠杆问题」vs「规则问题」。

- 若 BPTT-LoRA 也学不进（CE 不降）→ 杠杆/架构问题：旁路位置本身无足够影响力
- 若 BPTT-LoRA 能学（CE 明显降、recall>0）→ 规则问题：方向信息缺失是唯一瓶颈
  （则未来方向：更好的局部方向估计，或承认局部规则的边界）
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from snn_minilm import SpikingMiniLM
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS, gen_score
from stage3_t1_assoc import keyword_recall
from bpe_tokenizer import BPETokenizer


class LoRAModule(torch.nn.Module):
    """挂在 fc2 上的旁路（B 零初始化 + A 随机，标准 LoRA）。"""
    def __init__(self, d_out, d_in, r=8):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(r, d_in) * 0.05)
        self.B = torch.nn.Parameter(torch.zeros(d_out, r))

    def forward(self, x2):
        return (x2 @ self.A.T) @ self.B.T


def attach_lora(model, r=8):
    loras = {}
    hooks = []
    x2_store = {}
    for i, blk in enumerate(model.blocks):
        d_out, d_in = blk.fc2.weight.shape
        loras[i] = LoRAModule(d_out, d_in).to(next(model.parameters()).device)
        def mk_pre(idx):
            def hook(mod, inp):
                x2_store[idx] = inp[0]
            return hook
        def mk_post(idx):
            def hook(mod, inp, out):
                return out + loras[idx](x2_store[idx])
            return hook
        hooks.append(blk.fc2.register_forward_pre_hook(mk_pre(i)))
        hooks.append(blk.register_forward_hook(mk_post(i)))
    return loras, hooks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    x, y = slice_windows(tok, FACTS)
    print("[T1-BPTT-LoRA] windows=%d lr=%.1e r=%d dev=%s" % (x.shape[0], args.lr, args.r, dev))

    # 基线
    model0 = load_final(dev)
    ce0_list = []
    for i in range(x.shape[0]):
        with torch.no_grad():
            logits = model0(x[i].unsqueeze(0).to(dev))[0]
            mm = y[i].to(dev)[1:] != -100
            ce0_list.append(float(F.cross_entropy(logits[:-1][mm], y[i].to(dev)[1:][mm])))
    ce0 = sum(ce0_list) / len(ce0_list)
    tot_a, outs_a = gen_score(model0, tok, PROMPTS, FACTS)
    rec_a = sum(keyword_recall(t) for _, t, _ in outs_a)
    print("[A] mean fact-CE %.3f | recall %d | overlap %d" % (ce0, rec_a, tot_a))

    # BPTT-LoRA：主干冻结，仅旁路可训练
    model = load_final(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    loras, hooks = attach_lora(model, r=args.r)
    lora_params = [p for lora in loras.values() for p in lora.parameters()]
    npar = sum(p.numel() for p in lora_params)
    print("[BPTT-LoRA] trainable params=%d (%.2f%% of 34.3M)" % (npar, 100 * npar / 34.3e6))
    opt = torch.optim.AdamW(lora_params, lr=args.lr)

    curve = []
    nwin = x.shape[0]
    t0 = time.time()
    for k in range(args.steps):
        i = k % nwin
        logits = model(x[i].unsqueeze(0).to(dev))[0]
        mm = y[i].to(dev)[1:] != -100
        loss = F.cross_entropy(logits[:-1][mm], y[i].to(dev)[1:][mm])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if k % max(1, args.steps // 10) == 0 or k == args.steps - 1:
            with torch.no_grad():
                tot, _ = gen_score(model, tok, PROMPTS[:2], FACTS)
                curve.append((k, float(loss), tot))
            print("  win %3d | CE %.3f | overlap %d" % (k, float(loss), tot), flush=True)
    # 复测
    ce1_list = []
    for i in range(x.shape[0]):
        with torch.no_grad():
            logits = model(x[i].unsqueeze(0).to(dev))[0]
            mm = y[i].to(dev)[1:] != -100
            ce1_list.append(float(F.cross_entropy(logits[:-1][mm], y[i].to(dev)[1:][mm])))
    ce1 = sum(ce1_list) / len(ce1_list)
    for h in hooks:
        h.remove()
    tot_f, outs_f = gen_score(model, tok, PROMPTS, FACTS)
    rec_f = sum(keyword_recall(t) for _, t, _ in outs_f)
    sec = round(time.time() - t0, 1)

    lines = ["=== T1-BPTT-LoRA (steps=%d lr=%.1e r=%d dev=%s) ===" % (args.steps, args.lr, args.r, dev)]
    lines.append("FACTS mean CE: before %.3f -> after %.3f (delta %+.3f)" % (ce0, ce1, ce1 - ce0))
    lines.append("keyword recall: A=%d -> BPTT-LoRA=%d" % (rec_a, rec_f))
    lines.append("overlap: A=%d -> %d" % (tot_a, tot_f))
    for p, txt, ov in outs_f:
        lines.append("  %s (%d) -> %s" % (p, ov, txt[:70]))
    lines.append("--- curve (%.1fs) ---" % sec)
    lines.append(" | ".join("w%d:ce%.2f,ov%d" % c for c in curve))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_t1_bptt_lora_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved]", outp)


if __name__ == "__main__":
    main()
