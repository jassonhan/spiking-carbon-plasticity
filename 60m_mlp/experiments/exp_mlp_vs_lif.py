# -*- coding: utf-8 -*-
"""消融对照实验：LIF FFN → ReLU MLP，量化脉冲化的 loss 代价。

设计：
- MlpVariant = SpikingMiniLM 同构，唯一差异：FFN 中段 lif2 → ReLU（逐 token 递归消失）
- lif1 注意力门控保留（SymbolicLight 双路结构不动——单变量消融）
- warm start: snn34_dg7.best.pt（val 6.1506），fc1/fc2 权重直接复用（LIF 无参数）
- 训练配置与 SNN 周期完全一致：LR 5e-6 cosine 30000 步、batch 128、bf16 AMP、同数据混采
- 只跑 4000 步，与 SNN 周期的 eval 点（2000/4000）对齐比较
"""
import os
import sys
import math
import time
import numpy as np
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from snn_minilm import SpikingMiniLM, SpikingBlock   # noqa: E402
from bpe_tokenizer import BPETokenizer               # noqa: E402

DEV = "cuda"
BATCH, WIN = 128, 128
LR, LR_MIN, WARMUP, STEPS, RUN = 5e-6, 5e-7, 500, 30000, 4000
DATA = r"C:\wanjuan_local"
EXTRA = os.path.join(DATA, "wanjuan_extra_bpe.bin")
BEST = os.path.join(W, "SNN34_distill", "snn34_dg7.best.pt")
AMP = True


class MlpBlock(SpikingBlock):
    """同构 block，唯一差异：FFN 中段 lif2 脉冲 → ReLU 连续激活。"""

    def forward(self, x):
        a = self.attn(self.norm1(x))
        s = self.lif1(a)
        x = x + a * (0.5 + 0.5 * s)          # 注意力门控路径保持不变
        h = self.fc2(F.relu(self.fc1(self.norm2(x))))   # ← 唯一改动点
        return x + h


class MlpVariant(SpikingMiniLM):
    def __init__(self, vocab, dim, n_layers, n_heads, block_size, alpha=1.0):
        super().__init__(vocab, dim=dim, n_layers=n_layers, n_heads=n_heads,
                         block_size=block_size, alpha=alpha, gated_attn=True)
        self.blocks = torch.nn.ModuleList(
            [MlpBlock(dim, n_heads, block_size, 4, alpha=alpha, gated_attn=True)
             for _ in range(n_layers)])


tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
VOCAB = tok.vocab_size

train_tok = np.load(os.path.join(DATA, "clean_train_xl_bpe_cleaned.npy"), mmap_mode="r")
ex = np.memmap(EXTRA, dtype=np.int32, mode="r")
POOLS = [train_tok, ex]
WCOUNTS = [len(p) // WIN for p in POOLS]
PCUM = np.cumsum([c / sum(WCOUNTS) for c in WCOUNTS])
val_tok = np.load(os.path.join(DATA, "clean_val_bpe.npy"), mmap_mode="r")

# 三线探针切片（与训练脚本同款）
_seg = len(ex) // 16
_rng = np.random.default_rng(2026)
PROBES = {}
for pname, (a, b) in {"law": (3, 5), "news": (2, 3), "gen": (6, 16)}.items():
    lo, hi = a * _seg, min(b * _seg, len(ex) - WIN)
    starts = _rng.integers(lo, hi, size=32)
    PROBES[pname] = np.stack([np.asarray(ex[s:s + WIN], dtype=np.int64) for s in starts])

DIM, NLAY, NHED = 640, 10, 8
model = MlpVariant(VOCAB, DIM, NLAY, NHED, WIN, alpha=1.0).to(DEV)
ck = torch.load(BEST, map_location=DEV, weights_only=False)
sd = {k: v for k, v in ck["model"].items() if not k.endswith(".v")}
missing, unexpected = model.load_state_dict(sd, strict=False)
print("[model] warm-start %s | missing=%d unexpected=%d | params=%.1fM"
      % (BEST, len(missing), len(unexpected),
         sum(p.numel() for p in model.parameters()) / 1e6), flush=True)

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))


def get_lr(step):
    if step <= WARMUP:
        return LR * step / WARMUP
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR_MIN + (LR - LR_MIN) * 0.5 * (1.0 + math.cos(math.pi * prog))


def sample_batch():
    u = np.random.random()
    p = train_tok if u <= PCUM[0] else ex
    i = np.random.randint(0, len(p) // WIN) * WIN
    return np.asarray(p[i:i + BATCH * WIN], dtype=np.int64).reshape(BATCH, WIN)


@torch.no_grad()
def eval_ce(x):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, x.shape[0], 16):
        b = torch.from_numpy(x[i:i + 16]).to(DEV)
        logits = model(b)
        lo = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), b[:, 1:].reshape(-1))
        tot += lo.item() * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / n


vx = np.stack([np.asarray(val_tok[i * WIN:(i + 1) * WIN], dtype=np.int64) for i in range(32)])
print("[init] MLP-variant val=%.4f (SNN best=6.1506)  law/news/gen 将在 eval 打点" % eval_ce(vx), flush=True)

scaler = None
t0 = time.time()
for step in range(1, RUN + 1):
    lr = get_lr(step)
    for g in opt.param_groups:
        g["lr"] = lr
    x = torch.from_numpy(sample_batch()).to(DEV)
    if AMP:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            lo = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                                 x[:, 1:].reshape(-1))
    else:
        logits = model(x)
        lo = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    opt.zero_grad()
    lo.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

    if step % 100 == 0:
        tps = step * BATCH * WIN / max(1.0, time.time() - t0)
        print("step %d/%d | loss=%.4f lr=%.2e gnorm=%.2f | %.0fs | %.0f tok/s [mlp-variant]"
              % (step, RUN, lo.item(), lr, gn.item(), time.time() - t0, tps), flush=True)
    if step % 500 == 0:
        v = eval_ce(vx)
        pr = {k: eval_ce(x) for k, x in PROBES.items()}
        print("[eval] step %d val=%.4f (law=%.3f news=%.3f gen=%.3f)"
              % (step, v, pr["law"], pr["news"], pr["gen"]), flush=True)

print("MLP-RUN DONE")
