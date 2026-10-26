# -*- coding: utf-8 -*-
"""
Stage 3 路由破局 III：逐位置证据池化路由。

上一步结论（stage3_router_head.py）：cos-mean/center/LDA/线性头四种路由器
PROMPTS 全部 0.81 —— prompt 级 h 均值路由撞线性天花板，A↔D 混淆是表示问题。

假设：任务证据散布在个别 token 位置（"果园"→C、"上课"→D），整窗均值把
      证据稀释了。改为：逐位置 h 打分 + 位置间平均 logit → argmax。
拟合：全窗 + 后缀裁剪的逐位置 h（stride 采样控制规模）。
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS as PROMPTS_A
from stage3_t2_sequence import FACTS_B, PROMPTS_B
from stage3_stream4 import FACTS_C, PROMPTS_C, FACTS_D, PROMPTS_D
from stage3_disc_keys import HProbe
from stage3_router_head import train_head


def sample_positions(model, dev, tok, tasks, crop_min=24, crop_step=8, pos_stride=8):
    probe = HProbe(model, dev)
    fit, wins, prs = {}, {}, {}
    for name, facts, prompts in tasks:
        x, _ = slice_windows(tok, facts)
        hs, ws = [], []
        for i in range(x.shape[0]):
            ids = x[i].tolist()
            # 全窗逐位置
            xw = torch.tensor([ids], dtype=torch.long, device=dev)
            with torch.no_grad():
                model(xw)
            h = probe.h[0].float().cpu()
            hs.append(h[::pos_stride])                    # (T/stride, D)
            ws.append(h.mean(dim=0))
            # 裁剪逐位置
            for L in range(crop_min, 112, crop_step):
                xc = torch.tensor([ids[112 - L:]], dtype=torch.long, device=dev)
                with torch.no_grad():
                    model(xc)
                hs.append(probe.h[0].float().cpu()[::pos_stride])
        fit[name] = torch.cat(hs)                          # (N_pos, D)
        wins[name] = torch.stack(ws)
        # prompts：保留整条 h 序列（不均值），供逐位置打分
        pl = []
        for p in prompts:
            xp = torch.tensor([tok.encode(p)], dtype=torch.long, device=dev)
            with torch.no_grad():
                model(xp)
            pl.append(probe.h[0].float().cpu())
        prs[name] = pl
        print("[sample] %s: pos %d | wins %d | prompts %d" %
              (name, fit[name].shape[0], wins[name].shape[0], len(pl)), flush=True)
    return fit, wins, prs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=None)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--pos-stride", type=int, default=8)
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = __import__("bpe_tokenizer", fromlist=["BPETokenizer"]).BPETokenizer.load(
        os.path.join(W, "bpe_chinese_wanjuan.json"))
    tasks = [("A", FACTS, PROMPTS_A), ("B", FACTS_B, PROMPTS_B),
             ("C", FACTS_C, PROMPTS_C), ("D", FACTS_D, PROMPTS_D)]
    names = [t[0] for t in tasks]
    K = len(names)
    t0 = time.time()
    model = load_final(dev)
    fit, wins, prs = sample_positions(model, dev, tok, tasks, pos_stride=args.pos_stride)

    X = torch.cat([fit[t] for t in names])
    hbar = X.mean(dim=0)
    Xc = X - hbar
    y = torch.cat([torch.full((fit[t].shape[0],), i, dtype=torch.long)
                   for i, t in enumerate(names)])
    print("[train] 逐位置路由头 (%d pos 样本)..." % X.shape[0], flush=True)
    Wl, b = train_head(Xc, y, K, epochs=args.epochs)
    torch.save({"hbar": hbar, "W": Wl, "b": b, "names": names,
                "mode": "position"},
               os.path.join(W, "SNN34_distill", "stage3_router_pos.pt"))

    def head_any(h):
        """(T,D) 序列 -> 位置平均；(D,) 单向量 -> 直接打分。（线性等价于均值路由）"""
        lg = (h - hbar) @ Wl + b
        return lg.mean(dim=0) if lg.dim() == 2 else lg

    def post_pool(h):
        """(T,D) -> 逐位置 softmax 后验平均 -> log（非线性池化：证据不被单点淹没）。"""
        lg = ((h - hbar) @ Wl + b)
        if lg.dim() == 1:
            return lg
        return torch.log_softmax(lg, dim=1).mean(dim=0)

    def vote_pool(h):
        """(T,D) -> 逐位置 argmax 投票 -> 票数（多数决）。"""
        lg = ((h - hbar) @ Wl + b)
        if lg.dim() == 1:
            return lg
        votes = torch.zeros(lg.shape[1])
        top = lg.argmax(dim=1)
        for t in top:
            votes[t] += 1
        return votes

    def eval_router(fn, tag, seq=False):
        accs = {}
        for split, store in [("wins", wins), ("prompts", prs)]:
            hits, tot, marg, conf = 0, 0, [], {t: {} for t in names}
            for t in names:
                if split == "prompts":
                    items = store[t]
                else:
                    items = [store[t][i] for i in range(store[t].shape[0])]
                for h in items:
                    s = fn(h)
                    if s.dim() > 0:
                        top = int(s.argmax())
                        sv = s.sort(descending=True).values
                        m = float(sv[0] - sv[1])
                    else:
                        top, m = int(s.item()), 0.0
                    hits += int(names[top] == t)
                    tot += 1
                    marg.append(m)
                    conf[t][names[top]] = conf[t].get(names[top], 0) + 1
            accs[split] = (hits / tot, sum(marg) / len(marg), conf)
        return accs

    # 对照：同一把逐位置头，但 prompt 用均值（线性下与逐位置平均等价，仅作 sanity）
    rows = [
        ("pos-logitmean", eval_router(head_any, "plm")),
        ("pos-posterior", eval_router(post_pool, "pp")),
        ("pos-vote", eval_router(vote_pool, "pv")),
    ]
    lines = ["=== Stage3 逐位置证据池化路由 ==="]
    lines.append("拟合：全窗+裁剪逐位置 h (stride %d)，共 %d 样本" % (args.pos_stride, X.shape[0]))
    lines.append("%-16s %-26s %-26s" % ("路由器", "窗口集", "PROMPTS(诚实)"))
    for tag, accs in rows:
        w, p = accs["wins"], accs["prompts"]
        lines.append("%-16s acc %.2f marg %-8.3f | acc %.2f marg %-8.3f"
                     % (tag, w[0], w[1], p[0], p[1]))
    lines.append("")
    for tag, accs in rows:
        p = accs["prompts"]
        lines.append("%s 混淆: %s" % (tag, " | ".join(
            "%s→%s" % (t, ",".join("%s%d" % kv for kv in p[2][t].items())) for t in names)))
    fit_acc = float(((Xc @ Wl + b).argmax(1) == y).float().mean())
    lines.append("逐位置头拟合 acc %.3f" % fit_acc)
    lines.append("(%.0fs, dev=%s)" % (time.time() - t0, dev))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_router_pos_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
