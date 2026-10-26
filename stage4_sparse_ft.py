# -*- coding: utf-8 -*-
"""
Stage 4 pilot：稀疏表示端到端微调（SNN 主线正战场第一枪）。

理论动机：bolt-on pattern separation 判负（§6.11）——稀疏性必须由模型在训练中自己发育。
方法：34M final 起步，loss = CE + λ·L1(h_pre-ln)，短程微调（G 盘 wanjuan 14.7M tok 自监督），
      让 h 学出稀疏码；每臂结束复测任务路由天花板（raw 基线 0.81）。
臂：λ ∈ {0, 1e-4, 3e-4, 1e-3}，各 1500 步 batch16 lr2e-5。
指标：train/val CE（守语言能力，val 退化 >0.2 判废）、h 活跃率、PROMPTS 路由 acc。
对齐纪律：logits[:, :-1] vs y[:, 1:]（蒸馏恒等映射教训）。
"""
import os, sys, time, argparse, json, math
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from snn_minilm import SpikingMiniLM
from stage3_t1_fewshot import FACTS, PROMPTS as PROMPTS_A
from stage3_t2_sequence import FACTS_B, PROMPTS_B
from stage3_stream4 import FACTS_C, PROMPTS_C, FACTS_D, PROMPTS_D
from stage3_disc_keys import HProbe
from stage3_router_head import sample_all, train_head

BLOCK = 112


def load_model(dev):
    ck = torch.load(os.path.join(W, "SNN34_distill", "snn34_final.best.pt"),
                    map_location=dev, weights_only=False)
    cfg = ck["config"]
    m = SpikingMiniLM(cfg["vocab"], cfg["dim"], cfg["n_layers"], cfg["n_heads"],
                      cfg["block_size"], ff_mult=cfg.get("ff_mult", 4)).to(dev)
    sd = {k: v for k, v in ck["model"].items() if not k.endswith(".v")}
    m.load_state_dict(sd, strict=False)
    m.train()
    return m, cfg


def get_streams():
    """教师对 v4full（19408,112）——模型真正的语言分布（val 5.244 量测所在）。
    尾部 808 窗做监控（in-sample 警告：v4full 训练见过，仅用于臂间相对比较）。"""
    ti = np.load(os.path.join(W, "teacher_input_v4full.npy"))
    tl = np.load(os.path.join(W, "teacher_labels_v4full.npy"))
    return (ti[:-808], tl[:-808]), (ti[-808:], tl[-808:])


def batch_pair(data, bs, rng):
    ti, tl = data
    idx = rng.integers(0, ti.shape[0], size=bs)
    x = torch.tensor(ti[idx].astype(np.int64), dtype=torch.long)
    y = torch.tensor(tl[idx].astype(np.int64), dtype=torch.long)
    return x, y


@torch.no_grad()
def eval_ce(model, data, dev, n_win=384, bs=32):
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


def sparsity_stats(h_list):
    """h_list: list of (T,D) tensors（cpu）→ 活跃率统计。"""
    hs = torch.cat([h.flatten() for h in h_list])
    n = hs.numel()
    return float((hs.abs() > 0.05).float().mean()), \
           float((hs.abs() > 0.01).float().mean())


def finetune(lam, steps, bs, lr, data_tr, data_va, dev, eval_every=250):
    model, cfg = load_model(dev)
    h_store = {}
    model.ln_f.register_forward_pre_hook(
        lambda mod, inp: h_store.__setitem__("h", inp[0]))
    rng = np.random.default_rng(42)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    curve = []
    t0 = time.time()
    for step in range(1, steps + 1):
        x, y = batch_pair(data_tr, bs, rng)
        x, y = x.to(dev), y.to(dev)
        model.reset_states()
        logits = model(x)
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                             y[:, 1:].reshape(-1))
        h = h_store["h"]
        l1 = h.abs().mean()
        loss = ce + lam * l1
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % eval_every == 0 or step == 1:
            model.eval()
            vce = eval_ce(model, data_va, dev)
            model.train()
            af05, af001 = sparsity_stats([h_store["h"].detach().float().cpu()])
            curve.append((step, float(ce), vce, af05, af001))
            print("  [lam=%.0e] step %d train %.3f val %.3f act(>.05) %.3f act(>.01) %.3f (%.0fs)"
                  % (lam, step, ce, vce, af05, af001, time.time() - t0), flush=True)
    return model, curve


def routing_eval(model, dev, tok):
    """稀疏微调后模型的任务路由天花板复测。"""
    probe = HProbe(model, dev)
    fit, wins, prs = sample_all(model, dev, tok, TASKS_DEF)
    names = [t[0] for t in TASKS_DEF]
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
    return hits / tot, conf


TASKS_DEF = None


def main():
    global TASKS_DEF
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--lams", default="0,1e-4,3e-4,1e-3", help="逗号分隔的 λ 列表")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--dev", default=None)
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = __import__("bpe_tokenizer", fromlist=["BPETokenizer"]).BPETokenizer.load(
        os.path.join(W, "bpe_chinese_wanjuan.json"))
    TASKS_DEF = [("A", FACTS, PROMPTS_A), ("B", FACTS_B, PROMPTS_B),
                 ("C", FACTS_C, PROMPTS_C), ("D", FACTS_D, PROMPTS_D)]
    tr, va = get_streams()
    data_tr, data_va = (tr[0], tr[1]), (va[0], va[1])
    print("[data] teacher v4full: train %d win | monitor %d win" %
          (tr[0].shape[0], va[0].shape[0]), flush=True)

    lines = ["=== Stage4 稀疏微调 pilot（CE + λ·L1(h)）==="]
    lines.append("起步: snn34_final (val 5.244)；%d 步 batch%d lr%g" %
                 (args.steps, args.bs, args.lr))
    results = {}
    # λ=0 先测原模型基线（不训练，直接评）
    m0, _ = load_model(dev)
    m0.eval()
    vce0 = eval_ce(m0, data_va, dev)
    acc0, conf0 = routing_eval(m0, dev, tok)
    lines.append("")
    lines.append("[lam=0 (未微调基线)] 监控CE %.3f | PROMPTS routing %.2f" % (vce0, acc0))
    print("[baseline] mon %.3f routing %.2f" % (vce0, acc0), flush=True)
    del m0
    if dev == "cuda":
        torch.cuda.empty_cache()

    for lam in [float(s) for s in args.lams.split(",")]:
        model, curve = finetune(lam, args.steps, args.bs, args.lr, data_tr, data_va, dev)
        model.eval()
        vce = eval_ce(model, data_va, dev)
        acc, conf = routing_eval(model, dev, tok)
        results[lam] = (vce, acc, curve, conf)
        lines.append("")
        lines.append("[lam=%.0e] val CE %.3f (Δ%+.3f) | PROMPTS routing %.2f"
                     % (lam, vce, vce - vce0, acc))
        lines.append("  混淆: " + " | ".join(
            "%s→%s" % (t, ",".join("%s%d" % kv for kv in conf[t].items()))
            for t in "ABCD"))
        lines.append("  curve: " + " | ".join(
            "s%d:tr%.2f va%.2f act%.2f" % (c[0], c[1], c[2], c[4]) for c in curve))
        print("[lam=%.0e] done val %.3f routing %.2f" % (lam, vce, acc), flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    lines.append("")
    lines.append("判读: 监控CE 相对 lam=0 臂退化 <+0.2 为守住语言能力（注意监控集 in-sample，只看臂间相对）；")
    lines.append("routing 显著 >0.81 → 稀疏发育成功，pattern separation 可端到端学出（bolt-on 判负的镜像证明）。")
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage4_sparse_ft_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
