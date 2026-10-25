# -*- coding: utf-8 -*-
"""rate_samples.py -- SNN 生成质量人眼评分工具（命令行交互）

用法（工作目录 = 本目录）：
  "C:/Program Files/PyManager/python.exe" rate_samples.py            # 默认当前最佳 SNN 权重
  python rate_samples.py --ckpt SNN60_distill/snn60.best.pt          # 指定其他权重
  python rate_samples.py --n 6                                       # 生成 6 条样本

评分维度（每条样本）：
  流畅 f 1-5   读起来顺不顺（1=字符沙拉，3=能读但不通，5=通顺自然）
  切题 r 1-5   和 prompt 相关吗（1=完全无关，5=紧密接续）
  备注          可空，随手记问题（如"复读""乱码""突然像教师了"）

结果追加到 ratings.jsonl（每行一个 JSON：时间戳/权重/prompt/文本/分数）。
"""
import argparse, json, os, sys, time

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
import torch
from snn_minilm import SpikingMiniLM
from bpe_tokenizer import BPETokenizer

VOCAB = 8922
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPTS = ["从前有一座山", "春天来了", "我最喜欢的食物是", "小明对妈妈说",
           "今天放学后", "他打开门，"]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=None, help="直接指定权重（跳过菜单）")
ap.add_argument("--n", type=int, default=6, help="评分样本条数")
ap.add_argument("--temp", type=float, default=0.9)
ap.add_argument("--rp", type=float, default=1.3, help="repetition penalty（2026-10-19 A/B 实测有效压制复读）")
args = ap.parse_args()

# ---------- 版本菜单 ----------
MODELS = [
    ("SNN34 bootstrap最终(final)", "SNN34_distill/snn34_final.best.pt", "val 5.244 / 34M / v4-full末轮(rp=1.3)"),  # ← 新基线
    ("SNN34 蒸馏最佳(cont)", "SNN34_distill/snn34_cont.best.pt",        "val 5.879 / 34M / 教师蒸馏 27k步"),
    ("SNN34 蒸馏 phase1",       "SNN34_distill/snn34_distill.best.pt", "val 6.110 / 34M / 教师蒸馏 7k步"),
    ("SNN60 蒸馏",              "SNN60_distill/snn60.best.pt",         "val 6.003 / 60M / 教师蒸馏 20k步(LR失配轮)"),
    ("SNN34 wanjuan旧版",       r"I:\wanjuan_corpus\snn_34M_551M.best.pt", "50220步 / 34M / 551M语料自监督(需I盘)"),
]
avail = [(lab, p, d) for lab, p, d in MODELS if os.path.exists(p if os.path.isabs(p) else os.path.join(W, p))]

if args.ckpt:
    CKPT = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(W, args.ckpt)
    assert os.path.exists(CKPT), "checkpoint 不存在: " + CKPT
else:
    print("可选模型版本:")
    for i, (lab, p, d) in enumerate(avail, 1):
        print("  [%d] %-24s %s" % (i, lab, d))
    sel = input("选择 [1]: ").strip()
    idx = int(sel) - 1 if sel.isdigit() and 1 <= int(sel) <= len(avail) else 0
    CKPT = avail[idx][1] if os.path.isabs(avail[idx][1]) else os.path.join(W, avail[idx][1])
    print("已选: %s\n" % avail[idx][0])

OUT = os.path.join(W, "ratings.jsonl")

tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
ck = torch.load(CKPT, map_location=DEV, weights_only=False)
cfg = ck["config"]
model = SpikingMiniLM(cfg["vocab"], cfg["dim"], cfg["n_layers"], cfg["n_heads"],
                      cfg.get("block_size", 128), ff_mult=cfg.get("ff_mult", 4)).to(DEV)
_sd = model.state_dict()
_mm = [k for k in ck["model"] if k in _sd and _sd[k].shape != ck["model"][k].shape]
assert all(k.endswith(".v") for k in _mm), _mm
_fit = {k: t for k, t in ck["model"].items() if k not in _mm}
_res = model.load_state_dict(_fit, strict=False)
assert not _res.unexpected_keys
model.eval()

print("=" * 62)
print("SNN 生成质量人眼评分")
print("权重: %s | best=%.4f | device=%s" % (
    os.path.basename(CKPT), ck.get("best_val", ck.get("best_loss", -1)), DEV))
print("每条样本打两个分: 流畅 1-5 / 切题 1-5（直接回车 = 记 0 表示没感觉）")
print("=" * 62)

n = min(args.n, len(PROMPTS) * 2)
pick = [PROMPTS[i % len(PROMPTS)] for i in range(n)]
records = []
try:
    for i, p in enumerate(pick, 1):
        ids = torch.tensor(tok.encode_array(p), dtype=torch.long).to(DEV)
        model.reset_states()
        with torch.no_grad():
            out = model.generate(ids, 60, temperature=args.temp, top_k=40, rep_penalty=args.rp)
        text = tok.decode(out)
        print("\n[%d/%d] PROMPT: %s" % (i, n, p))
        print("-" * 62)
        print(text)
        print("-" * 62)
        f = input("流畅 f 1-5: ").strip()
        r = input("切题 r 1-5: ").strip()
        note = input("备注(可空): ").strip()
        records.append(dict(
            ts=time.strftime("%Y-%m-%d %H:%M"),
            ckpt=os.path.basename(CKPT),
            best_val=ck.get("best_val", ck.get("best_loss")),
            prompt=p, text=text,
            fluency=int(f) if f.isdigit() else 0,
            relevance=int(r) if r.isdigit() else 0,
            note=note))
except (KeyboardInterrupt, EOFError):
    print("\n[中断] 已录入 %d 条" % len(records))

if records:
    with open(OUT, "a", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    fs = [r["fluency"] for r in records if r["fluency"]]
    rs = [r["relevance"] for r in records if r["relevance"]]
    print("\n[saved] %d 条 -> %s" % (len(records), OUT))
    if fs:
        print("本次平均: 流畅 %.2f / 切题 %.2f" % (sum(fs)/len(fs), sum(rs)/len(rs)))
    # 历史汇总
    hist_f, hist_r, hist_n = [], [], 0
    if os.path.exists(OUT):
        for line in open(OUT, encoding="utf-8"):
            try:
                d = json.loads(line)
                hist_n += 1
                if d.get("fluency"): hist_f.append(d["fluency"])
                if d.get("relevance"): hist_r.append(d["relevance"])
            except Exception:
                pass
    if hist_n:
        print("历史累计: %d 条 | 流畅均值 %.2f | 切题均值 %.2f" % (
            hist_n, sum(hist_f)/max(len(hist_f),1), sum(hist_r)/max(len(hist_r),1)))
else:
    print("未录入任何评分。")
