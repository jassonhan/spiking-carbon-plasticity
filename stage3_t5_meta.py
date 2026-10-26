# -*- coding: utf-8 -*-
"""
Stage 3 T5 元可塑性：脚手架初始化实验（learn to learn 的权重层面形式）。

问题：新任务 E 的学习速度能否被已有任务经验（ΔW_A..D）加快？
生物类比：先发育编程后经验学习——旧突触结构给新学习提供脚手架（scaffold）。

三臂对照（E 的私有 delta 初始化）：
  arm-zero     零初始化（现行默认，基线）
  arm-scaffold γ * mean(ΔW_A..D)（经验脚手架，γ=0.1）
  arm-randnorm 与 scaffold 总范数匹配的随机初始化（分离「结构」与「范数」）
指标：E 学习曲线（CE@50 间隔）、pre-CE、steps-to-6.5、终 CE。
流程：A-D 各 800 步（快照 deltas）→ 三臂各从快照重置后学 E 800 步。
"""
import os, sys, time, copy, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.dirname(os.path.abspath(__file__))
from stage3_t1_fewshot import load_final, slice_windows, FACTS, PROMPTS as PROMPTS_A
from stage3_t2_sequence import FACTS_B
from stage3_stream4 import FACTS_C, FACTS_D
from stage3_gated_stream import GatedLMS

# 任务 E：交通/城市出行（全新主题，未在 A-D 出现）
FACTS_E = [
    "公交车沿着环线缓缓行驶，车窗外的梧桐树一排排向后退去。",
    "地铁站台上，人们排队候车，广播里播报着下一班列车的到达时间。",
    "骑共享单车穿过老城区，石板路在车轮下发出轻微的咯吱声。",
    "早高峰的高架桥上，车辆排成长龙，喇叭声此起彼伏。",
    "渡轮驶过江面，甲板上的乘客倚着栏杆看两岸的灯火。",
    "高铁飞驰过田野，油菜花在车窗外连成一片金黄色的海洋。",
    "出租车司机熟练地穿过小巷，避开了拥堵的主干道。",
    "夜班电车空荡荡的，车灯照亮了铺满落叶的铁轨。",
    "机场大巴驶出航站楼，行李架上塞满了旅客的箱子。",
    "人行横道的绿灯亮起，人流像潮水一样涌向马路对面。",
    "维修工人连夜检修隧道，头灯的光柱扫过湿漉漉的岩壁。",
    "摩托车快递员在车流中穿梭，保温箱在后座上微微晃动。",
    "索道缆车缓缓爬上山坡，脚下是成片的茶园和溪流。",
    "码头工人把集装箱吊上货轮，汽笛声在雾中传得很远。",
    "校车停在村口，孩子们背着书包蹦蹦跳跳地跑过来。",
    "清晨的菜市场的三轮车驮着新鲜的蔬菜驶向街市。",
]


def ce_curve(gl, name, x, y, steps, lr, log_every=50):
    curve, hit_step, pre = [], None, None
    for k in range(steps):
        i = k % x.shape[0]
        ce = gl.learn_step(name, x[i], y[i], lr)
        if k == 0:
            pre = ce
        if (k + 1) % log_every == 0:
            curve.append((k + 1, ce))
            if hit_step is None and ce <= 6.5:
                hit_step = k + 1
    fin, _ = gl.pool_ce(x, y)
    return curve, pre, hit_step, fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps-ad", type=int, default=800)
    ap.add_argument("--steps-e", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--dev", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps_ad = args.steps_e = 60
    dev = args.dev or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = __import__("bpe_tokenizer", fromlist=["BPETokenizer"]).BPETokenizer.load(
        os.path.join(W, "bpe_chinese_wanjuan.json"))
    t0 = time.time()
    data_ad = {}
    for name, facts in [("A", FACTS), ("B", FACTS_B), ("C", FACTS_C), ("D", FACTS_D)]:
        data_ad[name] = slice_windows(tok, facts)
    xE, yE = slice_windows(tok, FACTS_E)
    print("[data] E windows: %d" % xE.shape[0], flush=True)

    model = load_final(dev)
    gl = GatedLMS(model, dev)

    # ---- 1. 学 A-D，快照 deltas（带缓存，重跑不重复学习）----
    snap_p = os.path.join(W, "SNN34_distill", "stage3_t5_deltas.pt")
    if os.path.exists(snap_p) and not args.smoke:
        ck = torch.load(snap_p, map_location=dev, weights_only=False)
        gl.deltas = {k: v.to(dev) for k, v in ck.items()}
        print("[snap] loaded cached A-D deltas", flush=True)
    else:
        for name in ["A", "B", "C", "D"]:
            x, y = data_ad[name]
            for k in range(args.steps_ad):
                i = k % x.shape[0]
                gl.learn_step(name, x[i], y[i], args.lr)
            gl.set_key(name, x)
            print("[learn] %s done" % name, flush=True)
        if not args.smoke:
            torch.save({k: v.cpu() for k, v in gl.deltas.items()}, snap_p)

    # ---- 2. 三臂学 E ----
    mean_d = torch.stack([gl.deltas[t] for t in "ABCD"]).mean(dim=0)
    rand_d = torch.randn(mean_d.shape, generator=torch.Generator().manual_seed(7))
    rand_d = rand_d.to(mean_d.device)
    rand_d = rand_d * (args.gamma * mean_d.norm() / rand_d.norm())   # 范数匹配
    snapshot = copy.deepcopy(gl.deltas)

    arms = [("zero", torch.zeros_like(mean_d)),
            ("scaffold", args.gamma * mean_d),
            ("randnorm", rand_d)]
    lines = ["=== T5 元可塑性：脚手架初始化 (gamma=%.2f, lr=%.2f) ==="
             % (args.gamma, args.lr)]
    lines.append("E 窗口数 %d；对照：zero=现行默认 / scaffold=经验均值 / randnorm=范数匹配随机"
                 % xE.shape[0])
    results = {}
    for arm, init in arms:
        gl.deltas = copy.deepcopy(snapshot)
        gl.deltas["E"] = init.clone()
        curve, pre, hit, fin = ce_curve(gl, "E", xE, yE, args.steps_e, args.lr)
        results[arm] = (pre, curve, hit, fin)
        lines.append("")
        lines.append("[%s] pre-CE %.3f | final %.3f | steps-to-6.5 %s"
                     % (arm, pre, fin, hit if hit else ">%d" % args.steps_e))
        lines.append("  curve: " + " ".join("w%d:%.2f" % c for c in curve))
        print("[arm] %s final %.3f" % (arm, fin), flush=True)

    # ---- 3. 结论行 ----
    lines.append("")
    z, s, r = results["zero"], results["scaffold"], results["randnorm"]
    lines.append("判读:")
    lines.append("  scaffold vs zero 终CE: %+.3f | steps: %s vs %s"
                 % (s[3] - z[3], s[2] or "-", z[2] or "-"))
    lines.append("  randnorm vs zero 终CE: %+.3f（若 scaffold<zero 且 randnorm≈zero → 结构有效；"
                 "若 randnorm≈scaffold → 只是范数效应）" % (r[3] - z[3]))
    lines.append("(%.0fs, dev=%s)" % (time.time() - t0, dev))
    report = "\n".join(lines)
    outp = os.path.join(W, "SNN34_distill", "stage3_t5_meta_report.txt")
    with open(outp, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("[saved] %s" % outp)


if __name__ == "__main__":
    main()
