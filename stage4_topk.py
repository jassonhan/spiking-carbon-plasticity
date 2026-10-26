# -*- coding: utf-8 -*-
"""
Stage4 前向 k-sparse 实验：结构压力产生稀疏（L1 判惰性后的正攻）。

机制：ln_f 输入处 top-k 掩码（前向强制只留 |h| top-k% 维，梯度经掩码值回传），
      等效「数字 DG」/ 脉冲阈值发放。掩码作为 forward_pre_hook 注入（先掩码后探针，
      探针看到的就是稀疏码）。
臂：
  noFT-{25,10,5}%   不微调直接评测（稀疏本身能否改善 pattern separation）
  ft-{25,10,5}%     1000 步教师对微调（模型适应稀疏后）
指标：监控 CE（守住语言能力）、活跃率（应=目标 k%）、PROMPTS 路由 acc（raw 密集基线 0.81）。
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage4_sparse_ft import (load_model, get_streams, batch_pair, eval_ce,
                              routing_eval)
import stage4_sparse_ft
from stage3_t1_fewshot import FACTS, PROMPTS as PROMPTS_A
from stage3_t2_sequence import FACTS_B, PROMPTS_B
from stage3_stream4 import FACTS_C, PROMPTS_C, FACTS_D, PROMPTS_D


def topk_mask(t, frac):
    k = max(1, int(t.shape[-1] * frac))
    v, idx = torch.topk(t.abs(), k, dim=-1)
    m = torch.zeros_like(t, dtype=torch.bool)
    m.scatter_(-1, idx, True)
    return m


def set_sparsity(model, frac):
    """ln_f 输入掩码 hook（须先于探针注册）。返回移除函数。"""
    def mask_hook(mod, inp):
        t = inp[0]
        t.data.mul_(topk_mask(t.detach(), frac))
    h = model.ln_f.register_forward_pre_hook(mask_hook)
    return h.remove


def activity_of(model_probe_store):
    h = model_probe_store
    return float((h.abs() > 0.05).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--fracs", default="0.25,0.10,0.05")
    ap.add_argument("--dev", default=None)
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    import bpe_tokenizer
    tok = bpe_tokenizer.BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    data_tr, data_va = get_streams()
    stage4_sparse_ft.TASKS_DEF = [("A", FACTS, PROMPTS_A), ("B", FACTS_B, PROMPTS_B),
                                  ("C", FACTS_C, PROMPTS_C), ("D", FACTS_D, PROMPTS_D)]
    print("[data] teacher v4full: train %d | monitor %d" %
          (data_tr[0].shape[0], data_va[0].shape[0]), flush=True)

    lines = ["=== Stage4 前向 k-sparse（结构压力稀疏）==="]
    lines.append("教师对微调 %d 步 batch%d lr%g；密集基线 routing 0.81" %
                 (args.steps, args.bs, args.lr))
    t0 = time.time()

    for frac in [float(s) for s in args.fracs.split(",")]:
        tag = "%d%%" % int(frac * 100)
        # ---- noFT：不训练直接评 ----
        model, _ = load_model(dev)
        model.eval()
        rm = set_sparsity(model, frac)
        probe_store = {}
        model.ln_f.register_forward_pre_hook(
            lambda mod, inp: probe_store.__setitem__("h", inp[0].detach()))
        vce = eval_ce(model, data_va, dev)
        # 路由评测需要探针看见稀疏 h——把探针 hook 换成读稀疏值的方式：
        # routing_eval 内部自建 HProbe(后注册→看到掩码后的值)
        acc, conf = routing_eval(model, dev, tok)
        lines.append("")
        lines.append("[noFT-%s] 监控CE %.3f | 活跃率 %.3f | routing %.2f"
                     % (tag, vce, activity_of(probe_store["h"]), acc))
        lines.append("  混淆: " + " | ".join(
            "%s→%s" % (t, ",".join("%s%d" % kv for kv in conf[t].items())) for t in "ABCD"))
        print("[noFT-%s] ce %.3f routing %.2f" % (tag, vce, acc), flush=True)
        rm()
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

        # ---- ft：微调适应 ----
        model, _ = load_model(dev)
        model.train()
        rm = set_sparsity(model, frac)
        probe_store2 = {}
        model.ln_f.register_forward_pre_hook(
            lambda mod, inp: probe_store2.__setitem__("h", inp[0]))
        rng = np.random.default_rng(42)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        curve = []
        for step in range(1, args.steps + 1):
            x, y = batch_pair(data_tr, args.bs, rng)
            x, y = x.to(dev), y.to(dev)
            model.reset_states()
            logits = model(x)
            ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                 y[:, 1:].reshape(-1))
            opt.zero_grad()
            ce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 250 == 0 or step == 1:
                model.eval()
                vce = eval_ce(model, data_va, dev)
                model.train()
                act = activity_of(probe_store2["h"].detach().float().cpu())
                curve.append((step, float(ce), vce, act))
                print("  [ft-%s] step %d tr %.3f va %.3f act %.3f (%.0fs)"
                      % (tag, step, ce, vce, act, time.time() - t0), flush=True)
        model.eval()
        vce = eval_ce(model, data_va, dev)
        acc, conf = routing_eval(model, dev, tok)
        lines.append("")
        lines.append("[ft-%s] 监控CE %.3f | 活跃率 %.3f | routing %.2f"
                     % (tag, vce, activity_of(probe_store2["h"].detach().float().cpu()), acc))
        lines.append("  混淆: " + " | ".join(
            "%s→%s" % (t, ",".join("%s%d" % kv for kv in conf[t].items())) for t in "ABCD"))
        lines.append("  curve: " + " | ".join(
            "s%d:tr%.2f va%.2f act%.2f" % c for c in curve))
        print("[ft-%s] done ce %.3f routing %.2f" % (tag, vce, acc), flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    lines.append("")
    lines.append("判读: noFT routing>0.81 → 稀疏本身即 pattern separation（DG 立即生效）；")
    lines.append("noFT 差但 ft 恢复并超越 → 稀疏码需发育（SNN 训练中发育假说）；")
    lines.append("两者都 ≤0.81 → 512 维 top-k 太粗，需扩大维度再稀疏（真 DG 扩展）。")
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage4_topk_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
