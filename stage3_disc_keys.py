# -*- coding: utf-8 -*-
"""
Stage 3 路由破局实验：判别性任务键（LDA / QDA）。

问题：现行任务键 = 该任务窗 h 均值，跨任务 cos 差仅 ~0.07 —— h 空间里任务分布
严重重叠（缺 pattern separation），软门控跨任务污染，WTA 硬门控是"蒙对"。

方法：后缀裁剪增广键样本。每个任务窗切出 len 24..112 (step 8) 的后缀序列，
      每条单独前向取 h 均值 → 每任务 ~150 个键样本（A/B 各 ~160，C/D ~12）。
      在键样本分布上拟合四种路由器，用各任务 PROMPTS（独立文本、未参与拟合）
      做诚实评估：
        R0 cos-mean   现行基线：argmax cos(h, μ_t)
        R1 cos-center 全局中心化（去掉语言共同方向）后 cos
        R2 LDA-Maha   池内协方差白化马氏判别：argmax μ_tᵀΣ⁻¹h - ½μ_tᵀΣ⁻¹μ_t
        R3 QDA        每任务协方差 + 重收缩（二次判别）
      附加：协方差收缩系数扫描（λ 折内交叉验证选优）。
"""
import os, sys, time, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS as PROMPTS_A
from stage3_t2_sequence import FACTS_B, PROMPTS_B
from stage3_stream4 import FACTS_C, PROMPTS_C, FACTS_D, PROMPTS_D


class HProbe:
    """ln_f 输入 h 的探针。"""

    def __init__(self, model, dev):
        self.model, self.dev = model, dev
        self.h = None
        model.ln_f.register_forward_pre_hook(lambda mod, inp: self._set(inp[0].detach()))

    def _set(self, t):
        self.h = t

    def hmean(self, ids):
        """ids: list[int] -> 该序列的 h 均值 (D,) float cpu"""
        x = torch.tensor([ids], dtype=torch.long, device=self.dev)
        with torch.no_grad():
            self.model(x)
        return self.h[0].float().mean(dim=0).cpu()


def cov_shrink(X, lam):
    """X: (N,D) -> 收缩协方差 (D,D)：(1-λ)Σ + λ*mean(diag)*I"""
    Xc = X - X.mean(dim=0, keepdim=True)
    S = Xc.T @ Xc / max(1, X.shape[0] - 1)
    d = torch.diagonal(S).mean()
    return S + lam * d * torch.eye(S.shape[0])


def logdet_spd(M):
    return torch.linalg.cholesky(M).diagonal().log().sum().item()


def route_scores(h, mus, method, extra):
    """返回 (K,) 分数，越高越像该任务。"""
    K, D = mus.shape
    if method == "cos-mean":
        return F.cosine_similarity(mus, h.unsqueeze(0), dim=1)
    if method == "cos-center":
        mu_c = mus - extra["hbar"]
        hc = h - extra["hbar"]
        return F.cosine_similarity(mu_c, hc.unsqueeze(0), dim=1)
    if method == "LDA":
        Si = extra["Si"]
        return mus @ (Si @ h) - 0.5 * (mus * (Si @ mus.T).T).sum(dim=1)
    if method == "QDA":
        sc = torch.zeros(K)
        for t in range(K):
            d = h - mus[t]
            sc[t] = -0.5 * float(d @ extra["Si_list"][t] @ d) - 0.5 * extra["logdet"][t]
        return sc
    raise ValueError(method)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default=None)
    ap.add_argument("--crop-min", type=int, default=24)
    ap.add_argument("--crop-step", type=int, default=8)
    args = ap.parse_args()
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = __import__("bpe_tokenizer", fromlist=["BPETokenizer"]).BPETokenizer.load(
        os.path.join(W, "bpe_chinese_wanjuan.json"))
    tasks = [
        ("A", FACTS, PROMPTS_A),
        ("B", FACTS_B, PROMPTS_B),
        ("C", FACTS_C, PROMPTS_C),
        ("D", FACTS_D, PROMPTS_D),
    ]
    t0 = time.time()
    model = load_final(dev)
    probe = HProbe(model, dev)

    # ---- 1. 采样：每任务后缀裁剪键样本 + 窗 h 均值 + prompt h 均值 ----
    fit, wins, prs = {}, {}, {}
    for name, facts, prompts in tasks:
        x, _ = slice_windows(tok, facts)
        samples = []
        for i in range(x.shape[0]):
            ids = x[i].tolist()
            # 窗本身（len 112）
            samples.append(probe.hmean(ids))
            # 后缀裁剪
            for L in range(args.crop_min, 112, args.crop_step):
                samples.append(probe.hmean(ids[112 - L:]))
        fit[name] = torch.stack(samples)                 # (N_t, D)
        wins[name] = torch.stack([probe.hmean(x[i].tolist()) for i in range(x.shape[0])])
        prs[name] = torch.stack([probe.hmean(tok.encode(p)) for p in prompts])
        print("[sample] %s: fit %d crops | %d windows | %d prompts" %
              (name, fit[name].shape[0], wins[name].shape[0], prs[name].shape[0]), flush=True)

    names = [t[0] for t in tasks]
    mus = {t: fit[t].mean(dim=0) for t in names}
    hbar = torch.cat([fit[t] for t in names]).mean(dim=0)

    # ---- 2. 基线区分度：原始 cos / 中心化 cos 的类均值两两夹角 ----
    sep_lines = []
    for tag, f in [("原始", lambda a, b: float(F.cosine_similarity(a, b, dim=0))),
                   ("中心化", lambda a, b: float(F.cosine_similarity(a - hbar, b - hbar, dim=0)))]:
        cs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                cs.append((names[i] + names[j], f(mus[names[i]], mus[names[j]])))
        sep_lines.append("  %s: %s" % (tag, " ".join("%s %.3f" % p for p in cs)))

    # ---- 3. 收缩系数折内选优（留窗交叉验证，A/B 各窗、C/D 记录但样本不足）----
    def loo_acc(method, lam):
        """每任务：留出 k 号窗，用其余窗的裁剪重拟 μ（协方差按全量拟合，μ 重拟），
        用留出窗的 h 均值路由。C/D 只有 1 窗，跳过。"""
        hits, tot = 0, 0
        for t in names:
            xs = wins[t]
            if xs.shape[0] < 2:
                continue
            n_per = 1 + len(range(args.crop_min, 112, args.crop_step))  # 全窗 + 裁剪数
            for k in range(xs.shape[0]):
                # 每窗连续贡献 n_per 个样本，留出该窗的全部样本
                base = k * n_per
                mask = torch.ones(fit[t].shape[0], dtype=torch.bool)
                mask[base:base + n_per] = False
                m2t = fit[t][mask]
                mus2 = {u: (mus[u] if u != t else m2t.mean(dim=0)) for u in names}
                extra = make_extra(mus2, {u: (fit[u] if u != t else m2t) for u in names},
                                   method, lam)
                s = route_scores(xs[k], torch.stack([mus2[u] for u in names]),
                                 method, {**extra, "hbar": hbar})
                top = names[int(s.argmax())]
                hits += int(top == t)
                tot += 1
        return hits / max(1, tot), tot

    def make_extra(mus_, fit_, method, lam):
        if method == "LDA":
            Xall = torch.cat([fit_[t] - mus_[t] for t in names])
            S = cov_shrink(Xall, lam)
            return {"Si": torch.linalg.inv(S)}
        if method == "QDA":
            Si_l, ld = [], []
            for t in names:
                S = cov_shrink(fit_[t], lam)
                Si_l.append(torch.linalg.inv(S))
                ld.append(logdet_spd(S))
            return {"Si_list": Si_l, "logdet": ld}
        return {}

    METHODS = ["cos-mean", "cos-center", "LDA", "QDA"]
    best_lam = {}
    print("[cv] 留窗交叉验证选收缩系数...", flush=True)
    for m in ["LDA", "QDA"]:
        accs = {}
        for lam in [0.05, 0.1, 0.2, 0.3, 0.5]:
            acc, tot = loo_acc(m, lam)
            accs[lam] = acc
        best = max(accs, key=accs.get)
        best_lam[m] = best
        print("    %s: %s -> best λ=%.2f (acc %.2f, n=%d)" %
              (m, {k: round(v, 2) for k, v in accs.items()}, best, accs[best], tot), flush=True)

    # ---- 4. 终评：四种路由器，窗口集（LOO）+ PROMPTS 集（诚实）----
    lines = ["=== Stage3 判别性任务键（LDA/QDA 路由）==="]
    lines.append("拟合样本：后缀裁剪 len %d..112 step %d；评估集：任务窗 + PROMPTS（未见文本）"
                 % (args.crop_min, args.crop_step))
    lines.append("类均值区分度:")
    lines.extend(sep_lines)
    lines.append("")
    hdr = "%-11s %-10s %-10s %-10s" % ("路由器", "窗口LOO", "PROMPTS", "prompt边距")
    lines.append(hdr)
    for m in METHODS:
        lam = best_lam.get(m, 0.1)
        extra = make_extra(mus, fit, m, lam)
        acc_w, n_w = loo_acc(m, lam)
        # PROMPTS 诚实评估
        hits, tot, marg = 0, 0, []
        for t in names:
            M = torch.stack([mus[u] for u in names])
            for i in range(prs[t].shape[0]):
                s = route_scores(prs[t][i], M, m, {**extra, "hbar": hbar})
                top = int(s.argmax())
                hits += int(names[top] == t)
                tot += 1
                sv = s.sort(descending=True).values
                marg.append(float(sv[0] - sv[1]))
        lines.append("%-11s %-10.2f %-10.2f %-10.3f"
                     % (m, acc_w, hits / tot, sum(marg) / len(marg)))
    lines.append("")
    # 混淆明细（诚实集，cos-mean vs LDA）
    for m in ["cos-mean", "LDA"]:
        lam = best_lam.get(m, 0.1)
        extra = make_extra(mus, fit, m, lam)
        M = torch.stack([mus[u] for u in names])
        conf = {t: {} for t in names}
        for t in names:
            for i in range(prs[t].shape[0]):
                s = route_scores(prs[t][i], M, m, {**extra, "hbar": hbar})
                top = names[int(s.argmax())]
                conf[t][top] = conf[t].get(top, 0) + 1
        lines.append("%s PROMPTS 混淆: %s" % (m, " | ".join(
            "%s→%s" % (t, ",".join("%s%d" % kv for kv in conf[t].items())) for t in names)))
    lines.append("(%.0fs, dev=%s)" % (time.time() - t0, dev))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_disc_keys_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
