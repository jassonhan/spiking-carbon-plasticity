# -*- coding: utf-8 -*-
"""
Stage 3 门控生成评估：GatedLMS 学习四任务后，用门控自回归生成测 recall 保持。
键方案：多索引（每窗一个情境键，WTA over 全部索引，任务=索引属主）——仿海马多痕迹。
对照：昨日无门控流（生成全崩）与门控 CE 保持（B/C/D 0.000 漂移）。
"""
import os, sys, time, argparse, random
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS as PROMPTS_A
from stage3_t2_sequence import FACTS_B, PROMPTS_B, KEYWORDS_B
from stage3_stream4 import FACTS_C, PROMPTS_C, KEYWORDS_C, FACTS_D, PROMPTS_D, KEYWORDS_D
from stage3_t1_assoc import KEYWORDS, keyword_recall
from bpe_tokenizer import BPETokenizer


class GatedLMSMulti:
    """多索引键版 GatedLMS：每窗一个情境键，WTA over 全部索引。"""

    def __init__(self, model, dev):
        self.model = model
        self.dev = dev
        self.W = model.lm_head.weight.data          # (V,D) 冻结
        self.deltas = {}                            # task -> (V,D)
        self.indices = {}                           # task -> [key(D,)]
        self.h_store = {}
        self._last_logits = None
        self.hook = model.ln_f.register_forward_pre_hook(
            lambda mod, inp: self.h_store.__setitem__("h", inp[0].detach()))

    def _h(self):
        return self.h_store["h"][0].float()

    def ensure_task(self, name):
        if name not in self.deltas:
            self.deltas[name] = torch.zeros_like(self.W)
        self.indices.setdefault(name, [])

    def learn_step(self, name, x_ids, y_ids, lr, clamp=0.01):
        dW = self.deltas[name]
        logits_base = self.model(x_ids.unsqueeze(0).to(self.dev))[0]
        h = self._h()
        logits = logits_base + h @ dW.T
        tgt = y_ids.to(self.dev)
        lg, tg = logits[:-1], tgt[1:]
        mm = tg != -100
        ce_vec = F.cross_entropy(lg, tg, reduction="none")
        ce = float(ce_vec.detach().mean())
        probs = F.softmax(logits.detach().float()[:-1], dim=-1)
        onehot = torch.zeros_like(probs)
        onehot[mm, tg[mm]] = 1.0
        delta = probs - onehot
        T = x_ids.shape[0]
        per = torch.zeros(T, device=self.dev)
        idx = torch.arange(1, T, device=self.dev)[mm]
        per[idx] = ce_vec.detach()
        w = per / (per.mean() + 1e-6)
        delta_w = (delta * w[:-1].unsqueeze(1)).sum(dim=0)
        h_w = (h[:-1] * w[:-1].unsqueeze(1)).sum(dim=0)
        dw = -lr * torch.outer(delta_w, h_w)
        n_ref = self.W.norm(dim=1, keepdim=True)
        dn = dw.norm(dim=1, keepdim=True)
        scale = torch.clamp(clamp * n_ref / (dn + 1e-8), max=1.0)
        dW.add_(dw * scale)
        n_dw = dW.norm(dim=1, keepdim=True)
        n_cap = 0.5 * n_ref
        over = (n_dw > n_cap).squeeze(1)
        if over.any():
            dW[over] *= (n_cap[over] / n_dw[over])
        return ce

    def index_task(self, name, x_list, prefix_len=10):
        """多索引：每窗两个键——全窗 h 均值 + 前 prefix_len 字 h（与短 prompt 形态匹配）。"""
        self.ensure_task(name)
        with torch.no_grad():
            for wi, xi in enumerate(x_list):
                self.model(xi.unsqueeze(0).to(self.dev))
                h_all = self._h().mean(dim=0).cpu()
                self.indices[name].append(h_all)
                n_pre = min(prefix_len, xi.shape[0])
                # 前缀键：需要单独前向（膜电位从头开始）
                self.model.reset_states()
                self.model(xi[:n_pre].unsqueeze(0).to(self.dev))
                self.indices[name].append(self._h().mean(dim=0).cpu())
                self.model.reset_states()

    def route(self, x_ids):
        """WTA over 全部索引 → (task, cos)。"""
        if not self.indices:
            return None, 0.0
        h_mean = self._h().mean(dim=0)
        best_t, best_c = None, -2.0
        for t, keys in self.indices.items():
            K = torch.stack(keys).to(self.dev)
            c = F.cosine_similarity(K, h_mean.unsqueeze(0), dim=1)
            ci = float(c.max())
            if ci > best_c:
                best_t, best_c = t, ci
        return best_t, best_c

    def logits_gated(self, x_ids):
        """base + 路由到的任务 delta。"""
        logits = self.model(x_ids.unsqueeze(0).to(self.dev))[0]
        h = self._h()
        t, c = self.route(x_ids)
        if t is not None:
            logits = logits + h @ self.deltas[t].T
        return logits, t, c

    def pool_ce(self, x, y, true_task=None):
        tot, n, hit = 0.0, 0, 0
        with torch.no_grad():
            for i in range(x.shape[0]):
                logits, top, _ = self.logits_gated(x[i])
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

    def generate(self, prompt_ids, max_new=60, temperature=0.9, top_k=40, rp=1.3):
        """门控自回归生成。路由=批前向（完整注意力上下文）一次决定；
        生成=逐 token（模型既定模式），固定该任务 delta。
        注：单 token 前向的 h 与批前向不同域（attention 无历史 K/V），不能用 EMA 动态路由。"""
        ids = list(prompt_ids[: self.model.block_size])
        # ① 路由决定：批前向（rich cue 完整上下文）
        with torch.no_grad():
            self.model.reset_states()
            self.model(torch.tensor([ids], device=self.dev))
            h_win = self._h()                       # (T,D)
            route_key = h_win.mean(dim=0)
        t_route, best_c = None, -2.0
        if self.indices:
            for t, keys in self.indices.items():
                K = torch.stack(keys).to(self.dev)
                c = F.cosine_similarity(K, route_key.unsqueeze(0), dim=1)
                ci = float(c.max())
                if ci > best_c:
                    best_c, t_route = ci, t
        dW = self.deltas.get(t_route)
        # ② 生成：逐 token（膜电位连续），固定 delta
        self.model.reset_states()
        gen = []
        last = None
        for t_i in ids:
            out = self.model(torch.tensor([[t_i]], device=self.dev))
            last = out.detach()[0, -1].float()
        for _ in range(max_new):
            logits = last.clone()
            if dW is not None:
                h = self._h()[0]
                logits = logits + h @ dW.T
            for t_i in ids[-64:]:
                if logits[t_i] > 0:
                    logits[t_i] /= rp
                else:
                    logits[t_i] *= rp
            if temperature > 0:
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits[logits < v[-1]] = -float("inf")
                p = torch.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(p, 1))
            else:
                nxt = int(logits.argmax())
            gen.append(nxt)
            ids.append(nxt)
            out = self.model(torch.tensor([[nxt]], device=self.dev))
            last = out.detach()[0, -1].float()
        return gen, t_route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-per-task", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps_per_task = 60
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    tasks = [
        ("A", FACTS, PROMPTS_A, KEYWORDS),
        ("B", FACTS_B, PROMPTS_B, KEYWORDS_B),
        ("C", FACTS_C, PROMPTS_C, KEYWORDS_C),
        ("D", FACTS_D, PROMPTS_D, KEYWORDS_D),
    ]
    data = {n: slice_windows(tok, f) for n, f, _, _ in tasks}
    model = load_final(dev)
    gl = GatedLMSMulti(model, dev)
    rows = []
    t0 = time.time()

    for name, facts, prompts, kws in tasks:
        x, y = data[name]
        ce_pre, _ = gl.pool_ce(x, y)
        print("[stream] task %s arrives | pre-CE %.3f" % (name, ce_pre), flush=True)
        gl.ensure_task(name)
        for k in range(args.steps_per_task):
            i = k % x.shape[0]
            gl.learn_step(name, x[i], y[i], args.lr)
        gl.index_task(name, x)
        for t in gl.deltas:
            xt, yt = data[t]
            ce, gacc = gl.pool_ce(xt, yt, true_task=t)
            rows.append(("%s@%s" % (t, name), ce, gacc))
        print("  " + " | ".join("%s: CE %.3f acc %.2f" % r for r in rows
                                if r[0].endswith("@%s" % name)), flush=True)

    # 门控生成评估（流末权重，全部任务）
    lines = ["=== Gated-LMS generation eval (multi-index keys, lr=%.2f) ===" % args.lr]
    lines.append("%-10s %8s %6s" % ("task@phase", "CE", "acc"))
    for name, ce, g in rows:
        lines.append("%-10s %8.3f %6.2f" % (name, ce, g))
    learn_ce, final_ce = {}, {}
    for name, ce, g in rows:
        t, ph = name.split("@")
        if t == ph:
            learn_ce[t] = ce
        final_ce[t] = ce
    lines.append("")
    lines.append("生成保持——富线索模式补全（cue=任务窗口前40字, 动态路由, rp=1.3）:")
    from collections import Counter
    for name, facts, prompts, kws in tasks:
        x, y = data[name]
        recalls = []
        samples = []
        for wi in range(min(3, x.shape[0])):
            cue = x[wi][:40].tolist()
            gen, _ = gl.generate(cue, max_new=60)
            txt = tok.decode(gen)
            rec = keyword_recall(txt, kws)
            recalls.append(rec)
            # 路由分布
            cue_route = gl.route(torch.tensor(cue, device=dev))[0]
            samples.append(("cue[%d] %s" % (wi, tok.decode(cue)[:20]), txt, rec, cue_route))
        ce_now, gacc = gl.pool_ce(x, y, true_task=name)
        lines.append("[%s] recall %d/%d | CE %.3f acc %.2f"
                     % (name, sum(recalls), len(recalls), ce_now, gacc))
        for p, txt, rec, routed in samples:
            lines.append("  CUE: %s (route=%s, rec=%d)" % (p, routed, rec))
            lines.append("  GEN: %s" % txt[:100])
    mean_drift = sum(final_ce[t] - learn_ce[t] for t in learn_ce) / len(learn_ce)
    lines.append("")
    lines.append("平均 CE 漂移: %+.3f (无门控地板 +1.9)" % mean_drift)
    lines.append("(%.0fs)" % (time.time() - t0))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_gated_gen_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
