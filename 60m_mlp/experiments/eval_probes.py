# -*- coding: utf-8 -*-
"""领域探针评测：把 65GB extra bin 均匀切 16 段，各段测 CE + 特征词领域归类。

- 权重: SNN34_distill/snn34_dg7.best.pt (val 6.1506)
- 口径: 与训练 val_loss() 一致（fp32, shift-1 CE, 窗口 128）
- 段采样: 每段均匀抽 48 窗（比官方 32 窗多一点，噪声更小）
- 输出: 各段 CE + 与官方 val 的 delta + 段内解码文本的特征词领域判定
"""
import os
import sys
import re
import numpy as np
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from snn_minilm import SpikingMiniLM          # noqa: E402
from bpe_tokenizer import BPETokenizer        # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BIN = r"C:\wanjuan_local\wanjuan_extra_bpe.bin"
BEST = os.path.join(W, "SNN34_distill", "snn34_dg7.best.pt")
TOKJ = os.path.join(W, "bpe_chinese_wanjuan.json")
NSEG, NW = 16, 48       # 16 段 × 48 窗

tok = BPETokenizer.load(TOKJ)
VOCAB = tok.vocab_size
DIM, NLAY, NHED = 640, 10, 8

model = SpikingMiniLM(VOCAB, DIM, NLAY, NHED, 128, alpha=1.0, gated_attn=True).to(DEV)
ck = torch.load(BEST, map_location=DEV, weights_only=False)
sd = {k: v for k, v in ck["model"].items() if not k.endswith(".v")}
model.load_state_dict(sd, strict=False)
model.eval()
print("[model] loaded %s (val %.4f)" % (BEST, ck.get("best", 0)), flush=True)


def ce_windows(x):
    """x: (n,128) int64 -> 平均 CE（官方口径 shift-1）。"""
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, x.shape[0], 16):
            b = torch.from_numpy(x[i:i + 16]).to(DEV)
            logits = model(b)
            lo = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                                 b[:, 1:].reshape(-1), ignore_index=-100)
            tot += lo.item() * b.shape[0]
            cnt += b.shape[0]
    return tot / cnt


def seg_windows(ex, lo, hi, nwin):
    """段内均匀抽 nwin 个 128 窗（种子固定，可复现）。"""
    rng = np.random.default_rng(2026)
    starts = rng.integers(lo, hi - 128, size=nwin)
    out = np.stack([np.asarray(ex[s:s + 128], dtype=np.int64) for s in starts])
    return out


FEATURES = {
    "新闻": ["记者", "报道", "新闻网", " editor", "采访", "日电", "月", "日电"],
    "法律": ["原告", "被告", "判决", "合同", "律师", "法院", "条款", "仲裁"],
    "考试": ["下列", "选项", "考点", "答案", "解析", "习题", "试题"],
    "专利": ["专利", "所述", "实施例", "发明", "权利要求", "附图"],
    "网络": ["哈哈", "帖", "网友", "转发", "点赞", "评论", "直播"],
}


def classify(text):
    hits = {k: sum(text.count(w) for w in ws) for k, ws in FEATURES.items()}
    best = max(hits, key=hits.get)
    return best if hits[best] >= 2 else "通用/其他"


ex = np.memmap(BIN, dtype=np.int32, mode="r")
N = len(ex)
seg_len = N // NSEG
print("[bin] %d tok, seg=%d M tok" % (N, seg_len // 1000000), flush=True)

print("\n=== 领域探针: extra bin 16 段 ===")
results = []
for i in range(NSEG):
    lo, hi = i * seg_len, min((i + 1) * seg_len, N - 128)
    x = seg_windows(ex, lo, hi, NW)
    v = ce_windows(x)
    # 解码 2 窗做领域归类
    txt = tok.decode(x[0][:200].tolist()) + tok.decode(x[1][:200].tolist())
    dom = classify(txt)
    results.append((i, v, dom))
    print("seg %2d [%4.1fG-%4.1fG] CE=%.4f  %s" % (i, lo / 1e9, hi / 1e9, v, dom),
          flush=True)

# 官方 val 校准（与训练完全同款 32 窗）
val = np.load(os.path.join(W, "..", "wanjuan_local", "clean_val_bpe.npy"),
              mmap_mode="r") if os.path.exists(
    os.path.join(W, "..", "wanjuan_local", "clean_val_bpe.npy")) else \
    np.load(r"C:\wanjuan_local\clean_val_bpe.npy", mmap_mode="r")
vx = np.stack([np.asarray(val[i * 128:(i + 1) * 128], dtype=np.int64) for i in range(32)])
v_official = ce_windows(vx)
print("\n=== 官方 clean val: CE=%.4f（训练口径）===" % v_official)

# base train 池参考
tr = np.load(r"C:\wanjuan_local\clean_train_xl_bpe_cleaned.npy", mmap_mode="r")
rng = np.random.default_rng(7)
starts = rng.integers(0, len(tr) - 128, size=48)
tx = np.stack([np.asarray(tr[s:s + 128], dtype=np.int64) for s in starts])
print("base train 池抽样: CE=%.4f" % ce_windows(tx))

vals = np.array([r[1] for r in results])
print("\n=== 汇总 ===")
print("extra bin 段 CE: min=%.4f max=%.4f spread=%.4f" % (vals.min(), vals.max(), vals.max() - vals.min()))
worst = sorted(results, key=lambda r: -r[1])[:5]
best3 = sorted(results, key=lambda r: r[1])[:3]
print("CE 最高（学得最差）5 段:", [(i, round(v, 4), d) for i, v, d in worst])
print("CE 最低（学得最好）3 段:", [(i, round(v, 4), d) for i, v, d in best3])
