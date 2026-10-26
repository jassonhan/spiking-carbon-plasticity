# -*- coding: utf-8 -*-
"""
Stage3 扩程：8 任务流漂移验证（GatedLMS + 判别路由头，任务数从 4 → 8）。
目的：零漂移结论（4 任务 +0.004）在任务数翻倍后是否保持——机制规模化验证。
新任务 E/F/G/H 主题互异（交通/厨房/图书馆/医院），与 A-D 一起全排序流。
路由头：8 任务的裁剪增广键样本训练（同 stage3_router_head 协议）。
"""
import os, sys, time, argparse, copy
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage3_gated_stream import GatedLMS
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS as PROMPTS_A
from stage3_t2_sequence import FACTS_B, PROMPTS_B
from stage3_stream4 import FACTS_C, PROMPTS_C, FACTS_D, PROMPTS_D
from stage3_t1_assoc import KEYWORDS
from stage3_t2_sequence import KEYWORDS_B
from stage3_stream4 import KEYWORDS_C, KEYWORDS_D
from stage3_t5_meta import FACTS_E

FACTS_F = [
    "妈妈系上围裙，把青菜放进水槽里仔细清洗。",
    "油锅热了，葱花下锅的瞬间香味充满了整个厨房。",
    "面团在案板上被反复揉搓，慢慢变得光滑有弹性。",
    "高压锅发出嘶嘶的声响，排骨汤的香气从阀口飘出来。",
    "她把切好的水果摆成花形，放在盘子的正中央。",
    "蒸笼里冒着白气，包子皮变得晶莹透亮。",
    "爷爷的拿手菜是红烧肉，糖色熬得恰到好处。",
    "小妹偷偷用手指蘸了一点蛋糕糊，甜甜地笑了。",
    "砧板上刀声笃笃，土豆被切成整齐的细丝。",
    "汤太咸了，妈妈又添了半碗清水和两块豆腐。",
    "烤箱预热完成，曲奇饼的黄油香气渐渐弥漫开来。",
    "除夕的饺子全家一起包，父亲擀皮最快。",
]
PROMPTS_F = ["厨房里", "妈妈做饭时", "除夕那天", "烤箱预热后"]
KEYWORDS_F = ["厨房", "妈妈", "锅", "香", "面团", "饺子", "烤箱", "汤"]

FACTS_G = [
    "图书馆里安静得能听见翻书页的沙沙声。",
    "管理员把归还的书按编号放回高高的书架。",
    "他坐在靠窗的位子上，阳光正好落在摊开的书页上。",
    "借书卡上已经盖满了印章，这是今年第五十本。",
    "儿童区的地毯上，孩子们围着老师听绘本故事。",
    "期刊阅览室里，老人戴着老花镜研究棋谱。",
    "新书推荐架上摆着刚到的科幻小说系列。",
    "闭馆音乐响起时，她才恋恋不舍地合上书。",
    "自习室里座无虚席，考研的学生们埋头刷题。",
    "电子阅览室开通了，爷爷学着用屏幕查资料。",
    "角落的旧书义卖摊上，一本诗集只要两块钱。",
    "雨天午后，图书馆的落地窗蒙上了一层薄雾。",
]
PROMPTS_G = ["图书馆里", "管理员", "自习室", "闭馆时"]
KEYWORDS_G = ["书", "图书馆", "借", "阅读", "书架", "管理员", "自习", "闭馆"]

FACTS_H = [
    "诊所门口排着长队，护士正在给病人们量体温。",
    "医生仔细听了听他的心肺，说问题不大。",
    "药房的窗口前排着队，处方单从窗口递了进去。",
    "小男孩摔破了膝盖，护士轻柔地给他消毒包扎。",
    "体检报告出来了，各项指标都基本正常。",
    "输液室里很安静，只有点滴瓶偶尔的滴答声。",
    "牙医诊室里传来钻头的嗡嗡声，患者紧张地攥着扶手。",
    "出院那天，全家人都来接他，鲜花堆满了病房。",
    "医生叮嘱他按时吃药，一周后来复查。",
    "候诊区的电视播放着健康科普节目。",
    "急诊室深夜依然灯火通明，救护车呼啸而至。",
    "老中医搭着脉，慢悠悠地讲着养生之道。",
]
PROMPTS_H = ["诊所门口", "医生说", "急诊室", "出院那天"]
KEYWORDS_H = ["医生", "护士", "病人", "药", "诊", "体检", "输液", "医院"]

TASKS8 = [
    ("A", FACTS, PROMPTS_A, KEYWORDS),
    ("B", FACTS_B, PROMPTS_B, KEYWORDS_B),
    ("C", FACTS_C, PROMPTS_C, KEYWORDS_C),
    ("D", FACTS_D, PROMPTS_D, KEYWORDS_D),
    ("E", FACTS_E, ["公交车上", "地铁站台", "早高峰", "清晨的菜市场"], ["公交", "地铁", "车", "站台", "渡轮", "高铁"]),
    ("F", FACTS_F, PROMPTS_F, KEYWORDS_F),
    ("G", FACTS_G, PROMPTS_G, KEYWORDS_G),
    ("H", FACTS_H, PROMPTS_H, KEYWORDS_H),
]


class Router8:
    """8 任务判别路由头：裁剪增广拟合（同 stage3_router_head 协议）。"""

    def __init__(self, model, dev, tok, tasks, crop_min=24, crop_step=8):
        from stage3_disc_keys import HProbe
        probe = HProbe(model, dev)
        self.names = [t[0] for t in tasks]
        fit = {}
        for name, facts, prompts, kws in tasks:
            x, _ = slice_windows(tok, facts)
            samples = []
            for i in range(x.shape[0]):
                ids = x[i].tolist()
                samples.append(probe.hmean(ids))
                for L in range(crop_min, 112, crop_step):
                    samples.append(probe.hmean(ids[112 - L:]))
            fit[name] = torch.stack(samples)
            print("[r8-sample] %s: %d" % (name, fit[name].shape[0]), flush=True)
        X = torch.cat([fit[t] for t in self.names])
        y = torch.cat([torch.full((fit[t].shape[0],), i, dtype=torch.long)
                       for i, t in enumerate(self.names)])
        hbar = X.mean(dim=0)
        from stage3_router_head import train_head
        self.Wl, self.b = train_head(X - hbar, y, len(self.names), epochs=500)
        self.hbar = hbar
        self.mus = {t: fit[t].mean(dim=0) for t in self.names}

    def load_into(self, gl):
        dev = next(gl.model.parameters()).device
        gl.router = {"hbar": self.hbar.to(dev), "W": self.Wl.to(dev),
                     "b": self.b.to(dev), "names": self.names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-per-task", type=int, default=600)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--order-seed", type=int, default=None,
                    help="随机打乱任务顺序的种子（默认 A-H 原序）")
    args = ap.parse_args()
    if args.smoke:
        args.steps_per_task = 30
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = __import__("bpe_tokenizer", fromlist=["BPETokenizer"]).BPETokenizer.load(
        os.path.join(W, "bpe_chinese_wanjuan.json"))
    data = {}
    for name, facts, prompts, kws in TASKS8:
        data[name] = slice_windows(tok, facts)
        print("[data] %s: %d windows" % (name, data[name][0].shape[0]), flush=True)
    model = load_final(dev)
    gl = GatedLMS(model, dev)
    order = list(TASKS8)
    if args.order_seed is not None:
        import random as _rd
        _rd.Random(args.order_seed).shuffle(order)
        print("[order] %s" % "→".join(t[0] for t in order), flush=True)
    print("[router] 训练 8 任务路由头...", flush=True)
    r8 = Router8(model, dev, tok, TASKS8)
    r8.load_into(gl)
    # 路由头窗口集自检
    hit, tot = 0, 0
    for name, facts, prompts, kws in TASKS8:
        x, y = data[name]
        for i in range(x.shape[0]):
            gl.ensure_task(name)
    rows = []
    t0 = time.time()
    for name, facts, prompts, kws in order:
        x, y = data[name]
        gl.ensure_task(name)
        for k in range(args.steps_per_task):
            i = k % x.shape[0]
            gl.learn_step(name, x[i], y[i], args.lr)
        gl.set_key(name, x)
        for t in gl.deltas:
            xt, yt = data[t]
            ce, gacc = gl.pool_ce(xt, yt, true_task=t)
            rows.append(("%s@%s" % (t, name), ce, gacc))
        print("  [%s 学完] %s" % (name, " | ".join(
            "%s: CE %.3f acc %.2f" % r for r in rows if r[0].endswith("@%s" % name))),
            flush=True)

    learn_ce, final_ce = {}, {}
    for name, ce, g in rows:
        t, ph = name.split("@")
        if t == ph:
            learn_ce[t] = ce
        final_ce[t] = ce
    lines = ["=== Gated-LMS 8-task stream (判别路由头, lr=%.2f, order=%s) ==="
             % (args.lr, "→".join(t[0] for t in order))]
    for name, ce, g in rows:
        lines.append("%-10s %8.3f %6.2f" % (name, ce, g))
    lines.append("")
    lines.append("保持分析:")
    drifts = {}
    for t in learn_ce:
        d = final_ce[t] - learn_ce[t]
        drifts[t] = d
        lines.append("  %s: 学到 %.3f -> 流末 %.3f (漂移 %+.3f)" % (t, learn_ce[t], final_ce[t], d))
    mean_d = sum(drifts.values()) / len(drifts)
    lines.append("平均漂移: %+.3f（4 任务基线 +0.004；无门控地板 +1.9/任务）" % mean_d)
    lines.append("(%.0fs)" % (time.time() - t0))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_stream8_report%s.txt" %
                        ("" if args.order_seed is None else "_s%d" % args.order_seed))
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
