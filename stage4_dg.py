# -*- coding: utf-8 -*-
"""
Stage4 数字 DG：扩展→稀疏→回投影 的结构压力稀疏（过夜 10k 步适应）。

理论：top-k 同维判负（§stage4_topk）+ 随机扩展判负（§6.11）→ 组合拳：
      h(512) --W_dg 固定随机--> E=relu(2048) --top-k 10%--> z 稀疏码
      --W_back 固定随机--> h'(512) --> ln_f --> lm_head
      躯干在瓶颈约束下适应；路由直接用稀疏码 z（2048 维，pattern separation 主战场）。
对照臂：noDG 密集基线 / DG-noFT / DG-ft-10k。
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage4_sparse_ft import load_model, get_streams, batch_pair, eval_ce
from stage3_disc_keys import HProbe
from stage3_router_head import train_head
from stage3_t1_fewshot import slice_windows, FACTS, PROMPTS as PROMPTS_A
from stage3_t2_sequence import FACTS_B, PROMPTS_B
from stage3_stream4 import FACTS_C, PROMPTS_C, FACTS_D, PROMPTS_D


class DGBlock(nn.Module):
    """数字 DG：E=relu(x@Wd) -> top-k -> z -> x'=z@Wb；Wd/Wb 固定。"""

    def __init__(self, ln, d_in, d_exp, frac, seed=13):
        super().__init__()
        self.ln = ln
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("Wd", torch.randn(d_in, d_exp, generator=g) / d_in ** 0.5)
        self.register_buffer("Wb", torch.randn(d_exp, d_in, generator=g) / d_exp ** 0.5)
        self.frac = frac
        self.last_z = None

    def forward(self, x):
        e = F.relu(x @ self.Wd)
        k = max(1, int(e.shape[-1] * self.frac))
        v, idx = torch.topk(e, k, dim=-1)
        keep = torch.zeros_like(e, dtype=torch.bool).scatter_(-1, idx, True)
        z = e * keep
        self.last_z = z.detach()
        return self.ln(z @ self.Wb)


def set_dg(model, frac, d_exp=2048):
    dg = DGBlock(model.ln_f, model.ln_f.weight.shape[0], d_exp, frac).to(
        next(model.parameters()).device)
    model.ln_f = dg
    return dg


@torch.no_grad()
def dg_eval_ce(model, data, dev, n_win=384, bs=32):
    rng = np.random.default_rng(123)
    tot, n = 0.0, 0
    for _ in range(n_win // bs):
        x, y = batch_pair(data, bs, rng)
        x, y = x.to(dev), y.to(dev)
        model.reset_states()
        logits = model(x)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                             y[:, 1:].reshape(-1))
        tot += float(ce) * bs
        n += bs
    return tot / n


def dg_routing(dg, model, dev, tok, tasks):
    """路由直接用稀疏码 z（2048 维）：裁剪/窗/prompt 的 z 均值。"""
    names = [t[0] for t in tasks]
    probe = HProbe(model, dev)          # ln_f(=DGBlock) 的 forward_pre → x（稠密输入）
    # z 探针：直接读 dg.last_z（forward 后有效）
    def zmean(ids):
        x = torch.tensor([ids], dtype=torch.long, device=dev)
        model.reset_states()
        with torch.no_grad():
            model(x)
        return dg.last_z[0].float().mean(dim=0).cpu()
    fit, wins, prs = {}, {}, {}
    for name, facts, prompts in tasks:
        x, _ = slice_windows(tok, facts)
        samples = []
        for i in range(x.shape[0]):
            ids = x[i].tolist()
            samples.append(zmean(ids))
            for L in range(24, 112, 8):
                samples.append(zmean(ids[112 - L:]))
        fit[name] = torch.stack(samples)
        wins[name] = torch.stack([zmean(x[i].tolist()) for i in range(x.shape[0])])
        prs[name] = torch.stack([zmean(tok.encode(p)) for p in prompts])
    X = torch.cat([fit[t] for t in names])
    y = torch.cat([torch.full((fit[t].shape[0],), i, dtype=torch.long)
                   for i, t in enumerate(names)])
    hbar = X.mean(dim=0)
    Wl, b = train_head(X - hbar, y, len(names), epochs=500)
    hits, tot, conf = 0, 0, {t: {} for t in names}
    for t in names:
        for h in prs[t]:
            s = (h - hbar) @ Wl + b
            top = int(s.argmax())
            hits += int(names[top] == t)
            tot += 1
            conf[t][names[top]] = conf[t].get(names[top], 0) + 1
    # 死维率：z 拟合样本中从激活过的维度占比
    used = (X.abs() > 0.05).any(dim=0).float().mean()
    return hits / tot, conf, float(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--d-exp", type=int, default=2048)
    ap.add_argument("--dev", default=None)
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    import bpe_tokenizer
    tok = bpe_tokenizer.BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    tasks = [("A", FACTS, PROMPTS_A), ("B", FACTS_B, PROMPTS_B),
             ("C", FACTS_C, PROMPTS_C), ("D", FACTS_D, PROMPTS_D)]
    data_tr, data_va = get_streams()
    t0 = time.time()
    lines = ["=== Stage4 数字 DG（扩展%d → top-k %d%% → 回投影，躯干适应 %d 步）==="
             % (args.d_exp, int(args.frac * 100), args.steps)]

    # ---- DG noFT ----
    model, _ = load_model(dev)
    model.eval()
    dg = set_dg(model, args.frac, args.d_exp)
    vce = dg_eval_ce(model, data_va, dev)
    acc, conf, used = dg_routing(dg, model, dev, tok, tasks)
    lines.append("")
    lines.append("[DG-noFT] 监控CE %.3f | routing %.2f | z 维使用率 %.2f" % (vce, acc, used))
    lines.append("  混淆: " + " | ".join(
        "%s→%s" % (t, ",".join("%s%d" % kv for kv in conf[t].items())) for t in "ABCD"))
    print("[DG-noFT] ce %.3f routing %.2f used %.2f" % (vce, acc, used), flush=True)
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()

    # ---- DG ft ----
    model, _ = load_model(dev)
    model.train()
    dg = set_dg(model, args.frac, args.d_exp)
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
        if step % 500 == 0 or step == 1:
            model.eval()
            vce = dg_eval_ce(model, data_va, dev)
            model.train()
            act = float((dg.last_z.abs() > 0.05).float().mean())
            curve.append((step, float(ce), vce, act))
            print("  [DG-ft] step %d tr %.3f va %.3f act %.3f (%.0fs)"
                  % (step, ce, vce, act, time.time() - t0), flush=True)
    model.eval()
    vce = dg_eval_ce(model, data_va, dev)
    acc, conf, used = dg_routing(dg, model, dev, tok, tasks)
    lines.append("")
    lines.append("[DG-ft] 监控CE %.3f | routing %.2f | z 维使用率 %.2f" % (vce, acc, used))
    lines.append("  混淆: " + " | ".join(
        "%s→%s" % (t, ",".join("%s%d" % kv for kv in conf[t].items())) for t in "ABCD"))
    lines.append("  curve: " + " | ".join("s%d:tr%.2f va%.2f act%.2f" % c for c in curve))
    print("[DG-ft] done ce %.3f routing %.2f" % (vce, acc), flush=True)

    lines.append("")
    lines.append("判读: DG-ft routing 显著 >0.81 且 CE 恢复 → 扩展+稀疏的发育路线成立；")
    lines.append("仍 ≤0.81 → 表示重叠是数据问题（任务分布本身重叠），转教师语料扩程线。")
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage4_dg_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)
    # 保存适应后权重（可能有用）
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                "frac": args.frac, "d_exp": args.d_exp},
               os.path.join(W, "SNN34_distill", "stage4_dg_ft.pt"))
    print("[saved-ckpt] SNN34_distill/stage4_dg_ft.pt")


if __name__ == "__main__":
    main()
