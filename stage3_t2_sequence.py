# -*- coding: utf-8 -*-
"""
Stage 3 T2：序列记忆 A→B→A —— 遗忘与恢复曲线（Stage3_碳基可塑性路线.md T2）。

阶段：1) 学任务 A（团子事实）至平台  2) 学任务 B（豆豆事实）至平台
      3) 回测 A（纯测，零学习）——遗忘程度  4) 重学 A——savings（节省再学习）
规则：lm_head-LMS（轮8 突破规则）+ EWC 锚定 + best-recall 快照。
度量：各阶段末的任务 CE 与关键词 recall。
"""
import os, sys, time, argparse, random
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS as PROMPTS_A, gen_score
from stage3_t1_assoc import keyword_recall, KEYWORDS
from bpe_tokenizer import BPETokenizer

# 任务 B：豆豆（小狗）事实——与任务 A（团子，猫）领域相近（更真实的干扰）或相远
FACTS_B = [
    "豆豆是一只金色的小狗，它的尾巴总是摇个不停。",
    "豆豆最喜欢玩飞盘，小明一扔它就飞奔出去接住。",
    "每天清晨，豆豆会准时叫小明起床去散步。",
    "豆豆吃饭的时候狼吞虎咽，碗底舔得干干净净。",
    "豆豆会握手，只要小明伸出手它就把爪子放上去。",
    "下雨天豆豆不爱出门，趴在门口看着雨发呆。",
    "豆豆有一次追着自己的尾巴转了整整五分钟。",
    "小区的猫都不怕豆豆，它们经常一起晒太阳。",
    "豆豆洗澡的时候很乖，吹风机一响就自己跑进浴室。",
    "小明的书包上还留着豆豆的小牙印，那是它小时候啃的。",
    "豆豆晚上睡觉前一定要叼着它的小毯子才肯睡。",
    "中秋节那天，豆豆对着月亮叫了好久，像在唱歌。",
    "豆豆能听懂自己的名字，隔着两个楼道喊它都会跑来。",
    "小明教豆豆数数，它会用叫声的次数来回答。",
    "豆豆最喜欢坐车，头伸出车窗迎着风眯起眼睛。",
    "冬天豆豆的爪子容易裂，小明给它涂护手霜。",
    "豆豆在草地上打滚的时候，肚子朝天四脚朝天。",
    "家里的沙发角落是豆豆的秘密基地，藏着它的玩具。",
    "豆豆听到门铃声总是第一个冲到门口，比谁都快。",
    "小明说豆豆是世界上最乖的狗，豆豆好像真的听懂了。",
]
PROMPTS_B = ["豆豆最喜欢", "每天清晨", "豆豆会", "下雨天"]
KEYWORDS_B = ["豆豆", "飞盘", "散步", "握手", "毯子", "门铃", "爪子", "月亮"]


def eval_task(model, tok, x, y, prompts, facts, kws):
    ce_list = []
    for i in range(x.shape[0]):
        with torch.no_grad():
            logits = model(x[i].unsqueeze(0).to(next(model.parameters()).device))[0]
            mm = y[i].to(next(model.parameters()).device)[1:] != -100
            ce_list.append(float(F.cross_entropy(logits[:-1][mm], y[i].to(next(model.parameters()).device)[1:][mm])))
    ce = sum(ce_list) / len(ce_list)
    tot, outs = gen_score(model, tok, prompts, facts)
    rec = sum(keyword_recall(t, kws) for _, t, _ in outs)
    return ce, rec, tot, outs


def lms_phase(model, tok, x, y, steps, lr, decay, tag, log_every=200, replay=None):
    """一个 LMS 学习阶段（含 best-CE 快照，结束时恢复）。
    replay=(xr, yr, frac)：以 frac 概率重放另一任务窗口（海马重放抗遗忘）。"""
    W_head = model.lm_head.weight.data
    W0 = W_head.clone().cpu()
    h_store = {}
    hook = model.ln_f.register_forward_pre_hook(
        lambda mod, inp: h_store.__setitem__("h", inp[0].detach()))
    nwin = x.shape[0]
    rng = random.Random(42)
    best_ce, best_W, best_k = float("inf"), None, -1
    for k in range(steps):
        if replay and rng.random() < replay[2]:
            xr, yr, _ = replay
            i = rng.randrange(xr.shape[0])
            xi, yi = xr[i], yr[i]
            used_replay = True
        else:
            i = k % nwin
            xi, yi = x[i], y[i]
            used_replay = False
        logits = model(xi.unsqueeze(0).to(W_head.device))[0]
        tgt = yi.to(W_head.device)
        lg, tg = logits[:-1], tgt[1:]
        mm = tg != -100
        ce_vec = F.cross_entropy(lg, tg, reduction="none")
        ce = float(ce_vec.mean())
        probs = F.softmax(logits.detach().float()[:-1], dim=-1)
        onehot = torch.zeros_like(probs)
        onehot[mm, tg[mm]] = 1.0
        delta = probs - onehot
        T = xi.shape[0]
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
            W_head.sub_(decay * (W_head - W0.to(W_head.device)))
        if ce < best_ce and not used_replay:
            # 混合流下只对主任务窗记快照（重放窗的 CE 不代表主任务进度）
            best_ce, best_W, best_k = ce, W_head.clone(), k
        if k % log_every == 0 or k == steps - 1:
            print("    [%s] win %4d | CE %.3f" % (tag, k, ce), flush=True)
    hook.remove()
    if best_W is not None:
        W_head.copy_(best_W)
    print("    [%s] best CE %.3f @ %d (restored)" % (tag, best_ce, best_k), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stepsA", type=int, default=3000)
    ap.add_argument("--stepsB", type=int, default=3000)
    ap.add_argument("--stepsReA", type=int, default=800, help="重学 A 的步数（测 savings）")
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--decay", type=float, default=2e-3)
    ap.add_argument("--replay", type=float, default=0.0,
                    help="学 B 时重放 A 窗的概率（海马重放抗遗忘，0=关闭）")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.stepsA = args.stepsB = 300
        args.stepsReA = 100
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    xA, yA = slice_windows(tok, FACTS)
    xB, yB = slice_windows(tok, FACTS_B)
    model = load_final(dev)
    R = []

    print("[T2] phase 0: baseline")
    ceA0, recA0, ovA0, _ = eval_task(model, tok, xA, yA, PROMPTS_A, FACTS, KEYWORDS)
    ceB0, recB0, ovB0, _ = eval_task(model, tok, xB, yB, PROMPTS_B, FACTS_B, KEYWORDS_B)
    R.append(("baseline-A", ceA0, recA0, ovA0))
    R.append(("baseline-B", ceB0, recB0, ovB0))
    print("  A: CE %.3f rec %d | B: CE %.3f rec %d" % (ceA0, recA0, ceB0, recB0))

    print("[T2] phase 1: learn A (%d steps)" % args.stepsA)
    lms_phase(model, tok, xA, yA, args.stepsA, args.lr, args.decay, "learnA")
    ceA1, recA1, ovA1, _ = eval_task(model, tok, xA, yA, PROMPTS_A, FACTS, KEYWORDS)
    R.append(("after-A", ceA1, recA1, ovA1))
    print("  A: CE %.3f rec %d" % (ceA1, recA1))

    print("[T2] phase 2: learn B (%d steps, replay=%.0f%%)" % (args.stepsB, 100 * args.replay))
    rp = (xA, yA, args.replay) if args.replay > 0 else None
    lms_phase(model, tok, xB, yB, args.stepsB, args.lr, args.decay, "learnB", replay=rp)
    ceA2, recA2, ovA2, _ = eval_task(model, tok, xA, yA, PROMPTS_A, FACTS, KEYWORDS)
    ceB1, recB1, ovB1, _ = eval_task(model, tok, xB, yB, PROMPTS_B, FACTS_B, KEYWORDS_B)
    R.append(("after-B(A-test)", ceA2, recA2, ovA2))
    R.append(("after-B(B-test)", ceB1, recB1, ovB1))
    print("  A(遗忘测试): CE %.3f rec %d | B: CE %.3f rec %d" % (ceA2, recA2, ceB1, recB1))

    print("[T2] phase 3: relearn A (%d steps, savings test)" % args.stepsReA)
    lms_phase(model, tok, xA, yA, args.stepsReA, args.lr, args.decay, "relearnA", log_every=100)
    ceA3, recA3, ovA3, _ = eval_task(model, tok, xA, yA, PROMPTS_A, FACTS, KEYWORDS)
    R.append(("relearn-A", ceA3, recA3, ovA3))
    print("  A(恢复): CE %.3f rec %d" % (ceA3, recA3))

    lines = ["=== T2 sequence memory A->B->A (LMS lr=%.2f decay=%.1e dev=%s) ===" % (args.lr, args.decay, dev)]
    lines.append("%-16s %8s %6s %6s" % ("phase", "CE", "recall", "overlap"))
    for name, ce, rec, ov in R:
        lines.append("%-16s %8.3f %6d %6d" % (name, ce, rec, ov))
    # savings 分析：重学恢复的增益占首次学习增益的比例（标准 savings 口径）
    forgot = ceA2 - ceA1
    saved = (ceA0 - ceA3) / max(1e-6, (ceA0 - ceA1))
    lines.append("遗忘(A): CE +%+.3f | 重学恢复: CE %+.3f | savings=%.0f%%" % (forgot, ceA3 - ceA2, 100 * saved))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_t2_sequence_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved]", outp)


if __name__ == "__main__":
    main()
