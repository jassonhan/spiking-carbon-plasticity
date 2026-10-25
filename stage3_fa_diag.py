# -*- coding: utf-8 -*-
"""FA 方向质量诊断：FAPlasticizer 的 Δw 方向 vs BPTT 梯度方向的余弦（同窗对比）。"""
import os, sys
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from stage3_plastic import FAPlasticizer
from stage3_t1_fewshot import load_final, slice_windows, FACTS
from bpe_tokenizer import BPETokenizer


def cosim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    n = (a.norm() * b.norm()).item()
    return float((a @ b).item() / n) if n > 0 else float("nan")


def main():
    dev = "cpu"
    tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    x, y = slice_windows(tok, FACTS)
    print("[fa-diag] windows=%d" % x.shape[0])
    model = load_final(dev)
    pl = FAPlasticizer(model, eta=1.0, verbose=False)   # eta=1 纯方向测量（大小无关 cos）
    pl.attach()

    # snapshot weights, measure dw, restore
    w_snap = {i: {"fc1": model.blocks[i].fc1.weight.data.clone(),
                  "fc2": model.blocks[i].fc2.weight.data.clone()} for i in range(len(model.blocks))}

    rows = []
    for k in range(x.shape[0]):
        for i in range(len(model.blocks)):   # restore
            model.blocks[i].fc1.weight.data.copy_(w_snap[i]["fc1"])
            model.blocks[i].fc2.weight.data.copy_(w_snap[i]["fc2"])
        model.zero_grad(set_to_none=True)
        # --- FA dw ---
        logits = model(x[k].unsqueeze(0).to(dev))[0]           # (T,V)
        tgt = y[k].to(dev)
        # per-token surprise weight（同 T1）
        lg, tg = logits[:-1], tgt[1:]
        mm = tg != -100
        ce_vec = F.cross_entropy(lg, tg, reduction="none")
        T = x[k].shape[0]
        per = torch.zeros(T, device=dev)
        idx = torch.arange(1, T, device=dev)[mm]
        per[idx] = ce_vec[mm].detach()
        w = per / (per.mean() + 1e-6)
        pl.observe_error(logits.detach(), tgt, token_weight=w)
        # --- BPTT grad ---
        for p in model.parameters():
            p.requires_grad_(True)
        logits2 = model(x[k].unsqueeze(0).to(dev))
        loss = F.cross_entropy(logits2[0, :-1][mm], tg[mm])
        loss.backward()
        cs1 = cosim(model.blocks[0].fc1.weight.grad - model.blocks[0].fc1.weight.grad*0,
                    model.blocks[0].fc1.weight.data - w_snap[0]["fc1"])
        # dw per block
        c1s, c2s = [], []
        for i in range(len(model.blocks)):
            dw1 = model.blocks[i].fc1.weight.data - w_snap[i]["fc1"]
            dw2 = model.blocks[i].fc2.weight.data - w_snap[i]["fc2"]
            c1s.append(cosim(dw1, model.blocks[i].fc1.weight.grad))
            c2s.append(cosim(dw2, model.blocks[i].fc2.weight.grad))
        rows.append((loss.item(), sum(c1s)/len(c1s), sum(c2s)/len(c2s)))
        print("win %d | CE %.3f | FA-vs-grad cos: fc1 %+.4f fc2 %+.4f (mean over %d blocks)"
              % (k, loss.item(), sum(c1s)/len(c1s), sum(c2s)/len(c2s), len(c1s)), flush=True)
        model.zero_grad(set_to_none=True)
        for p in model.parameters():
            p.requires_grad_(False)

    import statistics as st
    print("\n=== FA 方向质量汇总（%d 窗）===" % len(rows))
    print("cos(fc1): mean %+.4f | median %+.4f" % (st.mean(r[1] for r in rows), st.median(r[1] for r in rows)))
    print("cos(fc2): mean %+.4f | median %+.4f" % (st.mean(r[2] for r in rows), st.median(r[2] for r in rows)))
    print("判读: |cos|>0.1 显著; ~0 无关。对照: 率基 Hebbian 是 +0.028 / +0.002")


if __name__ == "__main__":
    main()
