# -*- coding: utf-8 -*-
"""
Stage 3 破局实验：任务门控稀疏输出（海马索引式模块化）。
架构：底座 lm_head 冻结；每任务私有增量 ΔW_t（LMS 只更新自己那份）；
      推理时用「情境键」软门控选择增量（键=该任务窗口的 ln_f 输入均值）。
预测：四任务流漂移 ≈ 0（对照无门控地板 +3.3/+0.8/+3.1）。
"""
import os, sys, time, argparse, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS as PROMPTS_A
from stage3_t2_sequence import FACTS_B, PROMPTS_B, KEYWORDS_B
from stage3_stream4 import FACTS_C, PROMPTS_C, KEYWORDS_C, FACTS_D, PROMPTS_D, KEYWORDS_D
from stage3_t1_assoc import KEYWORDS


class GatedLMS:
    """任务门控 LMS：底座冻结 + 每任务私有 ΔW + 情境键软门控。"""

    def __init__(self, model, dev, temp=8.0):
        self.model = model
        self.dev = dev
        self.temp = temp
        self.W = model.lm_head.weight.data          # (V,D) 冻结底座
        self.deltas = {}                            # task -> (V,D)
        self.keys = {}                              # task -> (D,)
        self.h_store = {}
        self.hook = model.ln_f.register_forward_pre_hook(
            lambda mod, inp: h_set(self.h_store, inp[0].detach()))

    def _h(self):
        return self.h_store["h"][0].float()         # (T,D)

    def ensure_task(self, name):
        if name not in self.deltas:
            self.deltas[name] = torch.zeros_like(self.W)

    def learn_step(self, name, x_ids, y_ids, lr, clamp=0.01):
        """单窗 LMS：只更新任务 name 的私有增量（硬门控，闭环——前向含自身 delta）。"""
        self.ensure_task(name)
        dW = self.deltas[name]
        logits_base = self.model(x_ids.unsqueeze(0).to(self.dev))[0]  # (T,V) 底座（冻结）
        h = self._h()                                                  # (T,D)
        logits = logits_base + h @ dW.T                                # 闭环：前向含自身 delta
        tgt = y_ids.to(self.dev)
        lg, tg = logits[:-1], tgt[1:]
        mm = tg != -100
        ce_vec = F.cross_entropy(lg, tg, reduction="none")
        ce = float(ce_vec.detach().mean())
        probs = F.softmax(logits.detach().float()[:-1], dim=-1)
        onehot = torch.zeros_like(probs)
        onehot[mm, tg[mm]] = 1.0
        delta = probs - onehot                                      # (T-1,V)
        T = x_ids.shape[0]
        per = torch.zeros(T, device=self.dev)
        idx = torch.arange(1, T, device=self.dev)[mm]
        per[idx] = ce_vec.detach()
        w = per / (per.mean() + 1e-6)
        delta_w = (delta * w[:-1].unsqueeze(1)).sum(dim=0)          # (V,)
        h_w = (h[:-1] * w[:-1].unsqueeze(1)).sum(dim=0)             # (D,)
        dw = -lr * torch.outer(delta_w, h_w)                        # (V,D)
        # 钳制参考 = 底座 W 行范数（delta 零初始化，不能用自己的范数——会冻死）
        n_ref = self.W.norm(dim=1, keepdim=True)
        dn = dw.norm(dim=1, keepdim=True)
        scale = torch.clamp(clamp * n_ref / (dn + 1e-8), max=1.0)
        dW.add_(dw * scale)
        # 累计上限：delta 行范数 ≤ 50% 底座行范数（防无界增长）
        n_dw = dW.norm(dim=1, keepdim=True)
        n_cap = 0.5 * n_ref
        over = (n_dw > n_cap).squeeze(1)
        if over.any():
            dW[over] *= (n_cap[over] / n_dw[over])
        return ce

    def set_key(self, name, x_list):
        """任务学习完成后：键 = 该任务所有窗 h 均值的均值。"""
        self.ensure_task(name)
        hs = []
        with torch.no_grad():
            for xi in x_list:
                self.model(xi.unsqueeze(0).to(self.dev))
                hs.append(self._h().mean(dim=0))
        self.keys[name] = torch.stack(hs).mean(dim=0)
        self.deltas[name] = self.deltas[name]                       # 保持

    def gated_logits(self, x_ids, wta=True):
        """门控推理：base + ΔW_argmax h^T（WTA 硬门控=模式补全；wta=False 为软门控诊断）。
        返回 (logits, gate dict, argmax task)。"""
        logits = self.model(x_ids.unsqueeze(0).to(self.dev))[0]     # (T,V) 底座
        h = self._h()                                               # (T,D)
        if not self.keys:
            return logits, {}, None
        keys = torch.stack([self.keys[t] for t in self.keys])       # (K,D)
        h_mean = h.mean(dim=0)                                      # (D,)
        cos = F.cosine_similarity(keys, h_mean.unsqueeze(0), dim=1) # (K,)
        names = list(self.keys.keys())
        if wta:
            top = int(cos.argmax())
            add = h @ self.deltas[names[top]].T
            gate = {t: (1.0 if i == top else 0.0) for i, t in enumerate(names)}
            return logits + add, gate, names[top]
        g = torch.softmax(self.temp * cos, dim=0)
        add = torch.zeros_like(logits)
        for gi, t in zip(g, names):
            if float(gi) < 0.01:
                continue
            add += float(gi) * (h @ self.deltas[t].T)
        gate = {t: round(float(gi), 3) for t, gi in zip(names, g)}
        return logits + add, gate, names[int(g.argmax())]

    def pool_ce(self, x, y, true_task=None):
        """池均值 CE + WTA 指派准确率。"""
        tot, n, hit = 0.0, 0, 0
        with torch.no_grad():
            for i in range(x.shape[0]):
                logits, _, top = self.gated_logits(x[i])
                tg = y[i].to(self.dev)
                lg, tgg = logits[:-1], tg[1:]
                mm = tgg != -100
                if mm.sum() == 0:
                    continue
                tot += float(F.cross_entropy(lg[mm], tgg))
                n += 1
                if true_task is not None and top == true_task:
                    hit += 1
        return tot / max(1, n), hit / max(1, n)


def h_set(store, t):
    store["h"] = t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-per-task", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--temp", type=float, default=8.0, help="门控 softmax 温度")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps_per_task = 60
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = __import__("bpe_tokenizer", fromlist=["BPETokenizer"]).BPETokenizer.load(
        os.path.join(W, "bpe_chinese_wanjuan.json"))
    tasks = [
        ("A", FACTS, PROMPTS_A, KEYWORDS),
        ("B", FACTS_B, PROMPTS_B, KEYWORDS_B),
        ("C", FACTS_C, PROMPTS_C, KEYWORDS_C),
        ("D", FACTS_D, PROMPTS_D, KEYWORDS_D),
    ]
    data = {}
    for name, facts, prompts, kws in tasks:
        x, y = slice_windows(tok, facts)
        data[name] = (x, y, prompts, kws)
    model = load_final(dev)
    gl = GatedLMS(model, dev, temp=args.temp)
    rows = []
    t0 = time.time()

    for name, facts, prompts, kws in tasks:
        x, y, _, _ = data[name]
        ce_pre, _ = gl.pool_ce(x, y)
        print("[stream] task %s arrives | pre-CE %.3f" % (name, ce_pre), flush=True)
        gl.ensure_task(name)
        curve = []
        for k in range(args.steps_per_task):
            i = k % x.shape[0]
            ce = gl.learn_step(name, x[i], y[i], args.lr)
            if k % max(1, args.steps_per_task // 4) == 0 or k == args.steps_per_task - 1:
                curve.append("w%d:ce%.2f" % (k, ce))
        print("    [learn%s] %s" % (name, " | ".join(curve)), flush=True)
        gl.set_key(name, x)
        # 本阶段末：全部已见任务 CE + WTA 指派准确率
        for t in gl.deltas:
            xt, yt, _, _ = data[t]
            ce, gacc = gl.pool_ce(xt, yt, true_task=t)
            rows.append(("%s@%s" % (t, name), ce, gacc))
        print("  " + " | ".join("%s: CE %.3f acc %.2f" % r for r in rows
                                if r[0].endswith("@%s" % name)), flush=True)

    lines = ["=== Gated-LMS 4-task stream (lr=%.2f temp=%.1f) ===" % (args.lr, args.temp)]
    lines.append("%-10s %8s %6s" % ("task@phase", "CE", "gate"))
    for name, ce, g in rows:
        lines.append("%-10s %8.3f %6.2f" % (name, ce, g))
    lines.append("")
    lines.append("保持分析（学到时 vs 流末）:")
    learn_ce, final_ce = {}, {}
    for name, ce, g in rows:
        t, ph = name.split("@")
        if t == ph:
            learn_ce[t] = ce
        final_ce[t] = ce
    for t in "ABCD":
        if t in learn_ce:
            lines.append("  %s: 学到 %.3f -> 流末 %.3f (漂移 %+.3f)"
                         % (t, learn_ce[t], final_ce[t], final_ce[t] - learn_ce[t]))
    mean_drift = sum(final_ce[t] - learn_ce[t] for t in learn_ce) / len(learn_ce)
    lines.append("平均漂移: %+.3f (无门控地板 +1.9/任务, T2v2b顺序+重放 +3.3/+0.8/+3.1)" % mean_drift)
    lines.append("(%.0fs)" % (time.time() - t0))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_gated_stream_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
