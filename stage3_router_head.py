# -*- coding: utf-8 -*-
"""
Stage 3 路由破局 II：可训练线性路由头（判别性任务键的强形式）。

上一步（stage3_disc_keys.py）结论：
  - 类均值 cos 0.94 全高度重叠 = 共享「语言均值」主成分掩盖一切；
  - 中心化后 A↔B 反向（-0.98）→ 区分度其实存在，被共享成分压住；
  - 闭式 LDA：PROMPTS 0.75→0.81，但 A↔D 仍混（同为叙事句式）。

本步：中心化 h 上训练线性 softmax 路由头 W(D,K)+b（全批 Adam，只动路由器，
      不碰模型本体），并与 cos-mean / cos-center / LDA 同台对比。
产出：SNN34_distill/stage3_router.pt（hbar/W/b/names），供门控流与生成评估复用。
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS as PROMPTS_A
from stage3_t2_sequence import FACTS_B, PROMPTS_B
from stage3_stream4 import FACTS_C, PROMPTS_C, FACTS_D, PROMPTS_D
from stage3_disc_keys import HProbe, cov_shrink, logdet_spd, route_scores


def sample_all(model, dev, tok, tasks, crop_min=24, crop_step=8):
    probe = HProbe(model, dev)
    fit, wins, prs = {}, {}, {}
    for name, facts, prompts in tasks:
        x, _ = slice_windows(tok, facts)
        samples = []
        for i in range(x.shape[0]):
            ids = x[i].tolist()
            samples.append(probe.hmean(ids))
            for L in range(crop_min, 112, crop_step):
                samples.append(probe.hmean(ids[112 - L:]))
        fit[name] = torch.stack(samples)
        wins[name] = torch.stack([probe.hmean(x[i].tolist()) for i in range(x.shape[0])])
        prs[name] = torch.stack([probe.hmean(tok.encode(p)) for p in prompts])
        print("[sample] %s: fit %d | wins %d | prompts %d" %
              (name, fit[name].shape[0], wins[name].shape[0], prs[name].shape[0]), flush=True)
    return fit, wins, prs


def train_head(Xc, y, K, epochs=500, lr=5e-3, wd=1e-3, seed=0):
    """Xc: (N,D) 已中心化；y: (N,) 任务 id。返回 W(D,K), b(K)。"""
    torch.manual_seed(seed)
    D = Xc.shape[1]
    Wl = torch.zeros(D, K, requires_grad=True)
    b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([Wl, b], lr=lr, weight_decay=wd)
    for e in range(epochs):
        opt.zero_grad()
        logits = Xc @ Wl + b
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        if e % 100 == 0 or e == epochs - 1:
            acc = float((logits.argmax(1) == y).float().mean())
            print("    epoch %d loss %.4f fit-acc %.3f" % (e, float(loss), acc), flush=True)
    return Wl.detach(), b.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=None)
    ap.add_argument("--epochs", type=int, default=500)
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = __import__("bpe_tokenizer", fromlist=["BPETokenizer"]).BPETokenizer.load(
        os.path.join(W, "bpe_chinese_wanjuan.json"))
    tasks = [("A", FACTS, PROMPTS_A), ("B", FACTS_B, PROMPTS_B),
             ("C", FACTS_C, PROMPTS_C), ("D", FACTS_D, PROMPTS_D)]
    names = [t[0] for t in tasks]
    t0 = time.time()
    model = load_final(dev)
    fit, wins, prs = sample_all(model, dev, tok, tasks)

    X = torch.cat([fit[t] for t in names])
    hbar = X.mean(dim=0)
    Xc = X - hbar
    y = torch.cat([torch.full((fit[t].shape[0],), i, dtype=torch.long)
                   for i, t in enumerate(names)])
    K = len(names)
    mus = {t: fit[t].mean(dim=0) for t in names}

    # LDA 基线（λ=0.05，上一步最优）
    Xres = torch.cat([fit[t] - mus[t] for t in names])
    Si = torch.linalg.inv(cov_shrink(Xres, 0.05))

    print("[train] 线性路由头（中心化输入）...", flush=True)
    Wl, b = train_head(Xc, y, K, epochs=args.epochs)
    head = {"hbar": hbar, "W": Wl, "b": b, "names": names}
    torch.save(head, os.path.join(W, "SNN34_distill", "stage3_router.pt"))

    def head_scores(h):
        return (Xc @ Wl + b)[0] if False else (h - hbar) @ Wl + b

    def eval_router(fn, tag):
        accs = {}
        for split, store in [("wins", wins), ("prompts", prs)]:
            hits, tot, marg, conf = 0, 0, [], {t: {} for t in names}
            for t in names:
                for i in range(store[t].shape[0]):
                    s = fn(store[t][i])
                    top = int(s.argmax())
                    hits += int(names[top] == t)
                    tot += 1
                    sv = s.sort(descending=True).values
                    marg.append(float(sv[0] - sv[1]))
                    conf[t][names[top]] = conf[t].get(names[top], 0) + 1
            accs[split] = (hits / tot, sum(marg) / len(marg), conf)
        return accs

    rows = []
    rows.append(("cos-mean", eval_router(lambda h: route_scores(
        h, torch.stack([mus[u] for u in names]), "cos-mean", {}), "cos-mean")))
    rows.append(("cos-center", eval_router(lambda h: route_scores(
        h, torch.stack([mus[u] for u in names]), "cos-center", {"hbar": hbar}), "cos-center")))
    rows.append(("LDA", eval_router(lambda h: route_scores(
        h, torch.stack([mus[u] for u in names]), "LDA",
        {"Si": Si, "hbar": hbar}), "LDA")))
    rows.append(("router-head", eval_router(head_scores, "head")))

    lines = ["=== Stage3 可训练线性路由头 vs 闭式基线 ==="]
    lines.append("%-12s %-14s %-16s" % ("路由器", "窗口集", "PROMPTS(诚实)"))
    for tag, accs in rows:
        w = accs["wins"]
        p = accs["prompts"]
        lines.append("%-12s acc %.2f marg %.3f | acc %.2f marg %.3f"
                     % (tag, w[0], w[1], p[0], p[1]))
    lines.append("")
    for tag, accs in rows:
        p = accs["prompts"]
        lines.append("%s PROMPTS 混淆: %s" % (tag, " | ".join(
            "%s→%s" % (t, ",".join("%s%d" % kv for kv in p[2][t].items())) for t in names)))
    # 头的逐类 fit/泛化差距
    fit_acc = float(((Xc @ Wl + b).argmax(1) == y).float().mean())
    lines.append("")
    lines.append("路由头拟合 acc %.3f（训练=裁剪集，注意与 PROMPTS 泛化差 = 过拟合度）" % fit_acc)
    lines.append("(%.0fs, dev=%s)" % (time.time() - t0, dev))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_router_head_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
