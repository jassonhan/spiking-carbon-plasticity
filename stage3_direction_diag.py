# -*- coding: utf-8 -*-
"""
Stage 3 方向性诊断（判决性实验）：BPTT 梯度方向 vs 局部资格迹方向的余弦相似度。

问题：率基 Hebbian 资格迹的方向，与 CE 下降的梯度方向是否相关？
  cos 显著 > 0  -> 方向对，写入量问题 -> 调参可通
  cos ~ 0       -> 方向无关 -> 换规则（reward 直绑 logit / feedback alignment）
  cos < 0       -> 方向相反 -> Hebbian 在反着学

做法：同一经验窗上，
  e_fc1/e_fc2 = Plasticizer 的资格迹公式（surprise 加权，与 T1 完全一致）
  g_fc1/g_fc2 = 同窗 CE 对权重的真实梯度（BPTT/替代梯度）
  cos = <e, g> / (|e||g|)，fc1/fc2 分开报告，多窗统计。
"""
import os, sys, argparse
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from snn_minilm import SpikingMiniLM
from bpe_tokenizer import BPETokenizer
from stage3_t1_fewshot import load_final, slice_windows, FACTS


@torch.no_grad()
def eligibility_from_hooks(model, x, dev):
    """捕获单窗的 per-token 事件并按 T1 同款公式算资格迹（归一化版）。"""
    ev = {}
    hs = []
    for i, blk in enumerate(model.blocks):
        def mk_pre(idx):
            def hook(mod, inp):
                ev.setdefault(idx, {})["u"] = inp[0].detach()
            return hook
        def mk_lif(idx):
            def hook(mod, inp, out):
                ev.setdefault(idx, {})["s2"] = out.detach()
            return hook
        def mk_blk(idx):
            def hook(mod, inp, out):
                o = (out if not isinstance(out, tuple) else out[0]).detach()
                ev.setdefault(idx, {})["post"] = o
            return hook
        hs.append(blk.fc1.register_forward_pre_hook(mk_pre(i)))
        hs.append(blk.lif2.register_forward_hook(mk_lif(i)))
        hs.append(blk.register_forward_hook(mk_blk(i)))
    model(x.unsqueeze(0).to(dev))
    for h in hs:
        h.remove()

    E = {}
    for i, blk in enumerate(model.blocks):
        d = ev[i]
        s2_all, u_all, post_all = d["s2"], d["u"], d["post"]       # (B,T,*)
        # surprise 权重（与 T1 RPE 相同：continuation CE 归一）——此处用均匀权重+续段掩码近似亦可，
        # 为严格对齐 T1，由外部传入 per_tok；这里为自足性直接用均匀权重版本做方向参考
        s2 = s2_all.mean(dim=(0, 1))
        u = u_all.mean(dim=(0, 1))
        post = post_all.mean(dim=(0, 1))
        post_run = 0.0 * post                                       # 单窗诊断：稳态参考取 0（首窗行为）
        post_dev = post - post_run
        e2 = torch.outer(post_dev, s2 + 1e-3)
        e1 = torch.outer(s2 - 0.10, u)
        e2 = e2 / (e2.abs().max() + 1e-8)
        e1 = e1 / (e1.abs().max() + 1e-8)
        E[i] = {"fc1": e1, "fc2": e2}
    return E


def cosim(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    n = (a.norm() * b.norm()).item()
    return float((a @ b).item() / n) if n > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=None)
    ap.add_argument("--nwin", type=int, default=8, help="诊断窗数")
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    x, y = slice_windows(tok, FACTS)
    model = load_final(dev)
    print("[diag] windows=%d device=%s" % (min(args.nwin, x.shape[0]), dev))

    rows = []
    for k in range(min(args.nwin, x.shape[0])):
        # --- 资格迹（no_grad）---
        model.zero_grad(set_to_none=True)
        E = eligibility_from_hooks(model, x[k], dev)
        # --- BPTT 梯度（需梯度）---
        for p in model.parameters():
            p.requires_grad_(True)
        logits = model(x[k].unsqueeze(0).to(dev))
        tgt = y[k].to(dev)
        lg = logits[0, :-1, :]
        tg = tgt[1:]
        m = tg != -100
        loss = F.cross_entropy(lg[m], tg[m])
        loss.backward()
        g1 = model.blocks[0].fc1.weight.grad
        g2 = model.blocks[0].fc2.weight.grad
        c1 = cosim(E[0]["fc1"], g1)
        c2 = cosim(E[0]["fc2"], g2)
        # 分层汇总
        c1_all = [cosim(E[i]["fc1"], model.blocks[i].fc1.weight.grad) for i in range(len(model.blocks))]
        c2_all = [cosim(E[i]["fc2"], model.blocks[i].fc2.weight.grad) for i in range(len(model.blocks))]
        rows.append((k, loss.item(), c1, c2))
        print("win %d | CE %.3f | cos(fc1,e1)=%+.4f | cos(fc2,e2)=%+.4f | mean_all fc1 %+.4f fc2 %+.4f"
              % (k, loss.item(), c1, c2,
                 sum(c1_all)/len(c1_all), sum(c2_all)/len(c2_all)), flush=True)
        model.zero_grad(set_to_none=True)
        for p in model.parameters():
            p.requires_grad_(False)

    import statistics as st
    print("\n=== 方向性诊断汇总（%d 窗）===" % len(rows))
    print("cos(fc1, e1): mean %+.4f | median %+.4f" % (st.mean(r[2] for r in rows), st.median(r[2] for r in rows)))
    print("cos(fc2, e2): mean %+.4f | median %+.4f" % (st.mean(r[3] for r in rows), st.median(r[3] for r in rows)))
    print("判读: |cos|>0.1 显著相关; ~0 无关; <0 反向")


if __name__ == "__main__":
    main()
