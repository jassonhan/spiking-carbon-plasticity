# -*- coding: utf-8 -*-
"""
Stage 3 长程持续学习流：4 个任务陆续到达，LMS + EWC decay + 全历史重放。
模拟运营期真实形态：任务串行到达，每学一个新任务就对所有旧任务做重放。
度量：每阶段末所有已见任务的 CE——画"记忆保持曲线"。
"""
import os, sys, time, argparse, random
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
import torch.nn.functional as F
from stage3_t2_sequence import lms_phase, eval_task, FACTS_B, PROMPTS_B, KEYWORDS_B
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS as PROMPTS_A
from stage3_t1_assoc import KEYWORDS

# 任务 C：果园/季节
FACTS_C = [
    "秋天的果园里，苹果树上挂满了红彤彤的果子。",
    "爷爷带着小明去摘苹果，篮子很快就装满了。",
    "熟透的苹果从树上掉下来，滚到了草丛里。",
    "奶奶用果园里的苹果做了香喷喷的苹果派。",
    "冬天的时候，果树穿上了一件白雪做的外衣。",
    "春天来了，果园里的树都开出了粉白色的花。",
    "小蜜蜂在花丛中飞来飞去，忙着采蜜。",
    "到了夏天，树枝上结出了一个个青色的小果子。",
]
PROMPTS_C = ["秋天到了，", "果园里", "春天来了，", "爷爷带着"]
KEYWORDS_C = ["苹果", "果园", "春天", "蜜蜂"]

# 任务 D：学校/课堂
FACTS_D = [
    "教室里，同学们正在认真听老师讲课。",
    "黑板上的算术题，小明第一个举手回答。",
    "下课铃响了，大家跑到操场上做游戏。",
    "老师夸奖了值日生，教室被打扫得干干净净。",
    "图书角里有一本关于恐龙的书，大家都抢着看。",
    "美术课上，小红画了一座美丽的房子。",
    "运动会上，小明在跑步比赛中得了第二名。",
    "放学前，老师布置了今天的家庭作业。",
]
PROMPTS_D = ["上课的时候，", "教室里", "下课了，", "运动会上"]
KEYWORDS_D = ["老师", "教室", "同学", "作业"]


def quick_pool_ce(model, xp, yp):
    """池均值 CE（不含生成评估，快）。"""
    W_head = model.lm_head.weight.data
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(xp.shape[0]):
            logits = model(xp[i].unsqueeze(0).to(W_head.device))[0]
            tgt = yp[i].to(W_head.device)
            lg, tg = logits[:-1], tgt[1:]
            mm = tg != -100
            if mm.sum() == 0:
                continue
            tot += float(F.cross_entropy(lg[mm], tg[mm]))
            n += 1
    return tot / max(1, n)


def sleep_phase(model, seen, steps, lr, decay, tag="sleep"):
    """再巩固期（门控版）：全部已见任务池上纯重放低 lr 均衡。
    门控：池均值 CE 变差 OR 任何单任务恶化 >0.5 → 整体回滚（防牺牲刚学任务）。"""
    if steps <= 0 or not seen:
        return
    W_head = model.lm_head.weight.data
    W_pre = W_head.clone()
    xp = torch.cat([sx for _, sx, sy in seen], dim=0)
    yp = torch.cat([sy for _, sx, sy in seen], dim=0)
    before_pool = quick_pool_ce(model, xp, yp)
    before_task = {t: quick_pool_ce(model, xt, yt) for t, xt, yt in seen}
    h_store = {}
    hook = model.ln_f.register_forward_pre_hook(
        lambda mod, inp: h_store.__setitem__("h", inp[0].detach()))
    nwin = xp.shape[0]
    rng = random.Random(7)
    for k in range(steps):
        i = rng.randrange(nwin)
        logits = model(xp[i].unsqueeze(0).to(W_head.device))[0]
        tgt = yp[i].to(W_head.device)
        lg, tg = logits[:-1], tgt[1:]
        mm = tg != -100
        ce_vec = F.cross_entropy(lg, tg, reduction="none")
        probs = F.softmax(logits.detach().float()[:-1], dim=-1)
        onehot = torch.zeros_like(probs)
        onehot[mm, tg[mm]] = 1.0
        delta = probs - onehot
        T = xp[i].shape[0]
        per = torch.zeros(T, device=W_head.device)
        idx = torch.arange(1, T, device=W_head.device)[mm]
        per[idx] = ce_vec.detach()
        w = per / (per.mean() + 1e-6)
        delta_w = (delta * w[:-1].unsqueeze(1)).sum(dim=0)
        h = h_store["h"][0].float()
        h_w = (h[:-1] * w[:-1].unsqueeze(1)).sum(dim=0)
        dw = -lr * torch.outer(delta_w, h_w)
        n = W_head.norm(dim=1, keepdim=True)
        dn = dw.norm(dim=1, keepdim=True)
        scale = torch.clamp(0.01 * n / (dn + 1e-8), max=1.0)
        W_head.add_(dw * scale)
        if decay > 0:
            W_head.sub_(decay * (W_head - W_pre))
    hook.remove()
    after_pool = quick_pool_ce(model, xp, yp)
    after_task = {t: quick_pool_ce(model, xt, yt) for t, xt, yt in seen}
    worst = max(after_task[t] - before_task[t] for t in after_task)
    if after_pool > before_pool or worst > 0.5:
        W_head.copy_(W_pre)
        detail = " ".join("%s%+.2f" % (t, after_task[t] - before_task[t]) for t in sorted(after_task))
        print("    [%s] %d steps | pool %.3f->%.3f worst%+.2f REJECTED (%s)"
              % (tag, steps, before_pool, after_pool, worst, detail), flush=True)
    else:
        detail = " ".join("%s%+.2f" % (t, after_task[t] - before_task[t]) for t in sorted(after_task))
        print("    [%s] %d steps | pool %.3f->%.3f accepted (%s)"
              % (tag, steps, before_pool, after_pool, detail), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-per-task", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--decay", type=float, default=2e-3)
    ap.add_argument("--replay-frac", type=float, default=0.3,
                    help="对全部历史任务的合计重放概率")
    ap.add_argument("--sleep", type=int, default=0,
                    help="每任务学习后的再巩固步数（门控版，0=关闭）")
    ap.add_argument("--final-sleep", type=int, default=0,
                    help="全流结束后的整夜再巩固步数（低 lr 全池重放）")
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
    model = load_final(dev)
    seen = []          # [(x, y)] 已学任务
    rows = []          # 评估行
    t0 = time.time()

    for name, facts, prompts, kws in tasks:
        x, y = slice_windows(tok, facts)
        # 基线 CE（任务到达时，未学）
        ce_pre, _, _, _ = eval_task(model, tok, x, y, prompts, facts, kws)
        print("[stream] task %s arrives | pre-CE %.3f | history=%s"
              % (name, ce_pre, [s[0] for s in seen]), flush=True)

        # 学习本任务：重放概率摊到全部历史任务上（拼接历史窗作重放池）
        if seen:
            xr = torch.cat([sx for _, sx, sy in seen], dim=0)
            yr = torch.cat([sy for _, sx, sy in seen], dim=0)
            # 重放预算按任务数扩展：每个旧任务保底 per-task 份额（防稀释）
            frac_total = min(0.85, args.replay_frac * len(seen))
            lms_phase(model, tok, x, y, args.steps_per_task, args.lr, args.decay,
                      "learn%s" % name, replay=(xr, yr, frac_total))
            print("  (replay total %.0f%% over %d old tasks)" % (100 * frac_total, len(seen)), flush=True)
        else:
            lms_phase(model, tok, x, y, args.steps_per_task, args.lr, args.decay,
                      "learn%s" % name)
        seen.append((name, x, y))

        # 睡眠再巩固：全部已见任务均匀重放，低 lr 均衡（门控版）
        if args.sleep > 0:
            sleep_phase(model, seen, args.sleep, args.lr * 0.1, args.decay,
                        tag="sleep@%s" % name)

        # 本阶段末：评估全部已见任务
        detail = []
        for sname, sx, sy in seen:
            sp, sk, sf = None, None, None
            for tn, fts, prs, kk in tasks:
                if tn == sname:
                    sp, sk, sf = prs, kk, fts
            ce, rec, ov, _ = eval_task(model, tok, sx, sy, sp, sf, sk)
            detail.append((sname, ce, rec, ov))
        for sname, ce, rec, ov in detail:
            rows.append(("%s@%s" % (sname, name), ce, rec, ov))
        print("  " + " | ".join("%s: CE %.3f rec %d" % (s, c, r) for s, c, r, _ in detail), flush=True)

    # 整夜再巩固：全流结束后，对全部任务池做一次长时低 lr 重放（仿日间学习+夜间巩固）
    if args.final_sleep > 0 and seen:
        print("[final sleep] %d steps, pool=%d windows"
              % (args.final_sleep, sum(sx.shape[0] for _, sx, sy in seen)), flush=True)
        sleep_phase(model, seen, args.final_sleep, args.lr * 0.1, args.decay * 0.5,
                    tag="final-sleep")
        # 整夜后再评估全部任务
        for sname, sx, sy in seen:
            sp, sk, sf = None, None, None
            for tn, fts, prs, kk in tasks:
                if tn == sname:
                    sp, sk, sf = prs, kk, fts
            ce, rec, ov, _ = eval_task(model, tok, sx, sy, sp, sf, sk)
            rows.append(("final-%s" % sname, ce, rec, ov))
            print("  final %s: CE %.3f rec %d" % (sname, ce, rec), flush=True)

    # 报告
    lines = ["=== T-stream: 4-task continual learning (LMS lr=%.2f decay=%.1e replay=%.0f%%) ==="
             % (args.lr, args.decay, 100 * args.replay_frac)]
    lines.append("%-12s %8s %6s %6s" % ("task@phase", "CE", "recall", "overlap"))
    for name, ce, rec, ov in rows:
        lines.append("%-12s %8.3f %6d %6d" % (name, ce, rec, ov))
    lines.append("")
    lines.append("保持分析（学到时 CE vs 流末 CE）:")
    best_at_learn = {}
    for name, ce, rec, ov in rows:
        t = name.split("@")[0]
        if name.endswith(t):          # after-X 行里 X==task 即学到时
            best_at_learn[t] = ce
    final = {}
    for name, ce, rec, ov in rows:
        t = name.split("@")[0]
        final[t] = ce                 # 最后一次出现即流末
    for t in "ABCD":
        if t in best_at_learn and t in final:
            lines.append("  %s: 学到 %.3f -> 流末 %.3f (漂移 %+.3f)" % (t, best_at_learn[t], final[t], final[t] - best_at_learn[t]))
    lines.append("(%.0fs)" % (time.time() - t0))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_stream4_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
