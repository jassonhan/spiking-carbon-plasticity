# -*- coding: utf-8 -*-
"""
Stage 3 T1：少样本适应测试 —— 局部可塑性的第一场硬仗（Stage3_碳基可塑性路线.md T1）。

问题：学会一个新事实/新风格，局部规则（外挂可塑性）需要多少窗？
对照：对照组 A 无更新（纯推理，能力上限基线）
      实验组 B 固定 Hebbian（无奖励调制，观察纯 Hebbian 是否漂移）
      实验组 C R-STDP（奖励调制，正式运营模式）

素材：8 句全新简单事实句（V4 数据里不存在的内容），切窗后作为"经验流"。
判读：生成里目标内容的字符 3-gram 重叠（bias test 同款判分）+ CE 轨迹。

时间纪律：GPU 大跑由计划任务在允许窗执行；本脚本 --smoke 用 CPU 小样冒烟。
"""
import os, sys, time, argparse
import torch
import numpy as np

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from snn_minilm import SpikingMiniLM
from bpe_tokenizer import BPETokenizer
from stage3_plastic import Plasticizer, window_ce

RPE = True   # 预测误差驱动资格迹（Stage3 轮3机制升级）

# 新事实素材（不在任何训练数据里；简单句式，符合"先简单"的运营期输入策略）
# v2：加长经验流（v1 只有 8 句切 3 窗，300 步在循环同 3 窗——经验多样性不足）
FACTS = [
    "小明养了一只紫色的小猫，它每天早上都趴在窗台上晒太阳。",
    "小猫的名字叫团子，因为它睡觉的时候总是缩成一个圆球。",
    "团子最喜欢吃小鱼干，每次听到塑料袋的声音就会跑过来。",
    "小明每天放学回家的第一件事就是给团子添水添粮。",
    "周末的时候，小明会带着团子在院子里追蝴蝶玩。",
    "团子有一次爬到了树上，小明搬来梯子才把它抱下来。",
    "冬天来了，小明给团子的窝里铺了一件旧毛衣。",
    "团子夜里会轻轻跳上小明的床，蜷在脚边呼噜呼噜地睡。",
    "团子第一次见到雪的时候，吓得毛都炸开了，躲在小明身后。",
    "小明用纸箱给团子做了一个城堡，团子在里面钻来钻去。",
    "邻居家的小狗来串门，团子弓着背哈气，把它吓跑了。",
    "团子会自己开门把手，家里各个房间的门都关不住它。",
    "小明写作业的时候，团子总喜欢趴在台灯旁边看着他。",
    "有一次团子把毛线球滚到了沙发底下，找了整整一天。",
    "夏天太热，团子整天躺在地板的阴凉处，肚皮朝上睡觉。",
    "小明每天睡前都会摸摸团子的头，跟它说晚安。",
    "团子最怕打雷，每次雷雨天都会钻进小明的被窝里。",
    "小区里的流浪猫都认识团子，它们隔着栅栏互相蹭鼻子。",
    "团子的生日那天，小明给它买了一根猫条当蛋糕。",
    "长大后的团子胖了一圈，跳上窗台都要助跑两步。",
]

PROMPTS = ["小明养了一只", "团子最喜欢", "每天放学回家", "周末的时候"]


def slice_windows(tok, texts, target_len=112):
    """把事实句编码、拼接、滑窗切成 (N,112) 经验流（同蒸馏管线风格）。"""
    ids = []
    for t in texts:
        ids.extend(tok.encode_array(t).tolist())
    xs, ys = [], []
    for s in range(0, max(1, len(ids) - target_len), 24):
        w = ids[s:s + target_len]
        if len(w) < target_len:
            w = w + [0] * (target_len - len(w))
        xs.append(w)
        ys.append([tid if tid else -100 for tid in w])   # continuation=自身（教师窗约定）
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def ngram_overlap(gen_text, fact_texts, n=3):
    """目标内容的字符 3-gram 重叠数（bias test 同款判分）。"""
    def grams(s, n):
        s = "".join(c for c in s if "\u4e00" <= c <= "\u9fff")
        return set(s[i:i + n] for i in range(len(s) - n + 1))
    target = set()
    for t in fact_texts:
        target |= grams(t, n)
    g = grams(gen_text, n)
    return len(g & target)


@torch.no_grad()
def gen_score(model, tok, prompts, facts, rp=1.3):
    tot = 0
    outs = []
    for p in prompts:
        ids = torch.tensor(tok.encode_array(p), dtype=torch.long).to(next(model.parameters()).device)
        torch.manual_seed(42)
        out = model.generate(ids, 50, temperature=0.9, top_k=40, rep_penalty=rp)
        txt = tok.decode(out)
        ov = ngram_overlap(txt, facts)
        tot += ov
        outs.append((p, txt, ov))
    return tot, outs


def load_final(dev):
    ck = torch.load(os.path.join(W, "SNN34_distill", "snn34_final.best.pt"),
                    map_location=dev, weights_only=False)
    cfg = ck["config"]
    m = SpikingMiniLM(cfg["vocab"], cfg["dim"], cfg["n_layers"], cfg["n_heads"],
                      cfg["block_size"], ff_mult=cfg.get("ff_mult", 4)).to(dev)
    sd = {k: v for k, v in ck["model"].items() if not k.endswith(".v")}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def run_group(model, tok, x, y, group, steps, dev, log_every=5, eta=None, novelty=0.0):
    """一组实验：steps 窗的经验流。返回每 step 的 (ce, fact_overlap) 曲线与最终生成。"""
    curve = []
    pl = None
    if group != "A":
        pl = Plasticizer(model, eta=eta or (2e-2 if group == "C" else 1.5e-3),
                         reward_scale=1.0, verbose=(group == "C"))
        pl.attach()
    for k in range(steps):
        i = k % x.shape[0]
        ce, per_tok = window_ce(model, x[i].to(dev), y[i].to(dev), return_per_tok=RPE)
        if group != "A":
            if RPE and per_tok is not None:
                # surprise 权重：continuation 段 CE 归一（预测差的位置驱动写入）
                w = per_tok.clone()
                w = w / (w.mean() + 1e-6) if float(w.mean()) > 0 else w
                pl.observe(token_weight=w)
            else:
                pl.observe()           # 消费 window_ce forward 时 hook 捕获的脉冲事件
            nov = novelty if k < x.shape[0] else 0.0   # 第一轮经验流 = 探索期（新奇奖励）
            pl.set_reward(ce if np.isfinite(ce) else 6.0, novelty=nov)
        if k % log_every == 0 or k == steps - 1:
            tot, _ = gen_score(model, tok, PROMPTS[:2], FACTS)
            curve.append((k, ce if np.isfinite(ce) else 6.0, tot))
            print("  [%s] win %3d | CE %.3f | fact-overlap %d" % (group, k, ce, tot), flush=True)
    if pl:
        pl.detach()
    tot, outs = gen_score(model, tok, PROMPTS, FACTS)
    return curve, tot, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60, help="经验窗数（少样本预算）")
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true", help="CPU 小样冒烟")
    ap.add_argument("--novelty", type=float, default=0.5, help="探索期新奇奖励加成（0=关）")
    ap.add_argument("--eta", type=float, default=None, help="覆盖 C 组学习率")
    args = ap.parse_args()

    dev = args.dev or ("cuda" if torch.cuda.is_available() and not args.smoke else "cpu")
    tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
    model = load_final(dev)
    x, y = slice_windows(tok, FACTS)
    print("[T1] windows=%d | device=%s | fact windows each %d tok" % (x.shape[0], dev, x.shape[1]))

    results = {}
    for group in ("A", "B", "C"):
        if group != "A":
            model = load_final(dev)          # 每组从同一初始权重重开
        t0 = time.time()
        curve, tot, outs = run_group(model, tok, x, y, group, args.steps, dev,
                                     log_every=max(1, args.steps // 8),
                                     eta=args.eta, novelty=args.novelty)
        results[group] = dict(curve=curve, overlap=tot, outs=outs, sec=round(time.time() - t0, 1))
        print("  [%s] final fact-overlap=%d (%.1fs)" % (group, tot, results[group]["sec"]), flush=True)

    lines = ["=== T1 few-shot adaptation (facts=%d, steps=%d, dev=%s) ===" % (len(FACTS), args.steps, dev)]
    for g in ("A", "B", "C"):
        lines.append("[%s] final overlap=%d | %.1fs" % (g, results[g]["overlap"], results[g]["sec"]))
        for p, txt, ov in results[g]["outs"]:
            lines.append("  %s (%d) -> %s" % (p, ov, txt[:80]))
    # 曲线对比
    lines.append("--- CE/overlap curves ---")
    for g in ("A", "B", "C"):
        lines.append("[%s] " % g + " | ".join("w%d:ce%.2f,ov%d" % (k, ce, ov) for k, ce, ov in results[g]["curve"]))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_t1_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved]", outp)


if __name__ == "__main__":
    main()
