# -*- coding: utf-8 -*-
"""
Stage4 任务感知表示训练：SupCon 对比损失直攻路由天花板（0.81）。

理论：稀疏四连判负 → 纯 CE 压力不产生 pattern separation；本实验给表示加
      「任务感知」目标——同一任务的窗 h 拉近、跨任务推远（SupCon，τ=0.1），
      CE 同时守住语言能力。评测仍用未见 PROMPTS（泛化，非记忆）。
批构造：每步 8 教师窗（CE）+ 8 任务窗（对比），单次前向单次反传。
臂：λc ∈ {0.3, 1.0, 3.0} × 1000 步。指标：监控 CE（教师分布守门）、PROMPTS routing。
判读：routing 显著 >0.81 → 表示可被任务感知目标塑造（正解）；仍 0.81 →
      A↔D 混淆是语料级歧义，转语料多样性线。
"""
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage4_sparse_ft import load_model, get_streams, batch_pair, eval_ce, routing_eval
import stage4_sparse_ft
from stage3_t1_fewshot import slice_windows
from stage3_zoo import TASKS16


def supcon(z, labels, tau=0.1):
    """z: (B,D) 已 L2 归一化；labels: (B,) 任务 id。SupCon 损失。"""
    sim = z @ z.T / tau                                   # (B,B)
    B = z.shape[0]
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(eye, -1e4)                           # -inf 会在 ×0 时产生 NaN
    pos = labels.unsqueeze(0) == labels.unsqueeze(1)      # (B,B)
    pos.masked_fill_(eye, False)
    has_pos = pos.any(dim=1)
    if not has_pos.any():
        return torch.tensor(0.0, device=z.device)
    logp = torch.log_softmax(sim, dim=1)
    loss = -(logp * pos.float()).sum(dim=1)[has_pos] / pos.sum(dim=1)[has_pos]
    return loss.mean()


def task_batch(zoo_x, zoo_label, bs_task, rng):
    idx = rng.integers(0, len(zoo_x), size=bs_task)
    x = torch.tensor(np.stack([zoo_x[i] for i in idx]).astype(np.int64), dtype=torch.long)
    t = torch.tensor([zoo_label[i] for i in idx], dtype=torch.long)
    return x, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--bs-task", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lams", default="0.3,1.0,3.0")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 10
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    import bpe_tokenizer
    tok = bpe_tokenizer.BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    stage4_sparse_ft.TASKS_DEF = [(t[0], t[1], t[2]) for t in TASKS16]
    data_tr, data_va = get_streams()

    # 任务窗池：16 主题任务的窗 + 任务标签
    zoo_x, zoo_label = [], []
    label_map = {}
    for ti, (name, facts, prompts, kws) in enumerate(TASKS16):
        label_map[name] = ti
        x, _ = slice_windows(tok, facts)
        zoo_x.extend([x[i].numpy() for i in range(x.shape[0])])
        zoo_label.extend([ti] * x.shape[0])
    zoo_x = np.stack(zoo_x)
    zoo_label = np.array(zoo_label)
    print("[data] teacher 18600 win | zoo %d win, %d tasks"
          % (len(zoo_x), len(TASKS16)), flush=True)

    lines = ["=== Stage4 任务感知表示训练（CE + λc·SupCon, tau=%.2f）===" % args.tau]
    lines.append("批: %d 教师窗(CE) + %d 任务窗(SupCon)，%d 步 lr%g；"
                 "密集基线 routing 0.81（4 任务）/ zoo 待测" % (args.bs, args.bs_task, args.steps, args.lr))

    for lam_s in args.lams.split(","):
        lam = float(lam_s)
        model, _ = load_model(dev)
        model.train()
        h_store = {}
        model.ln_f.register_forward_pre_hook(
            lambda mod, inp: h_store.__setitem__("h", inp[0]))
        rng = np.random.default_rng(42)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        curve = []
        t0 = time.time()
        for step in range(1, args.steps + 1):
            # 教师窗（CE）
            xt, yt = batch_pair(data_tr, args.bs, rng)
            xt, yt = xt.to(dev), yt.to(dev)
            model.reset_states()
            logits_t = model(xt)
            ce = F.cross_entropy(logits_t[:, :-1].reshape(-1, logits_t.shape[-1]).float(),
                                 yt[:, 1:].reshape(-1))
            # 任务窗（SupCon）
            xz, tz = task_batch(zoo_x, zoo_label, args.bs_task, rng)
            xz = xz.to(dev)
            model.reset_states()
            model(xz)
            hz = h_store["h"].float().mean(dim=1)             # (B_task,D)
            zn = F.normalize(hz, dim=1)
            lc = supcon(zn, tz.to(dev), tau=args.tau)
            loss = ce + lam * lc
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 250 == 0 or step == 1:
                model.eval()
                vce = eval_ce(model, data_va, dev)
                model.train()
                curve.append((step, float(ce), float(lc), vce))
                print("  [lam=%.1f] step %d ce %.3f supcon %.3f mon %.3f (%.0fs)"
                      % (lam, step, ce, lc, vce, time.time() - t0), flush=True)
        model.eval()
        vce = eval_ce(model, data_va, dev)
        acc, conf = routing_eval(model, dev, tok)
        lines.append("")
        lines.append("[lam=%.1f] 监控CE %.3f | 16任务 PROMPTS routing %.2f" % (lam, vce, acc))
        lines.append("  混淆: " + " | ".join(
            "%s→%s" % (t, ",".join("%s%d" % kv for kv in conf[t].items()))
            for t in sorted(conf.keys())))
        lines.append("  curve: " + " | ".join(
            "s%d:ce%.2f sc%.2f mon%.2f" % c for c in curve))
        print("[lam=%.1f] done mon %.3f routing %.2f" % (lam, vce, acc), flush=True)
        torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "lam": lam, "steps": args.steps},
                   os.path.join(W, "SNN34_distill", "stage4_contrast_lam%s.pt" % lam_s))
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    lines.append("")
    lines.append("判读: routing 显著>0.81 → 表示可被任务感知目标塑造；")
    lines.append("监控 CE 相对 6.08 的退化幅度 = 对比损失的语言代价。")
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage4_contrast_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
