# -*- coding: utf-8 -*-
"""dg7-MLP 长周期预训练：LIF FFN → ReLU 的正式主线（消融实验胜出后转正）。

与 snn_pretrain_dg7.py 对齐：
- 数据混采（base 550M + extra 163 亿按窗口数加权）、batch 128、bf16 AMP
- LR 5e-6 cosine、WARMUP 500、clip 1.0
- 三线 val（law/news/gen，种子 2026）
- 存档：BEST（val 改善时）+ LATEST（每 2000 步）→ SNN34_distill/snn34_mlp{.best}.pt
- 差异：无早停（跑满 STEPS 以确认平台）；MLP 无 LIF 递归（无膜电位状态）
- 日志：SNN34_distill/_mlp_launch.log（追加）
"""
import os
import sys
import math
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from snn_minilm import SpikingMiniLM, SpikingBlock   # noqa: E402
from bpe_tokenizer import BPETokenizer               # noqa: E402

STEPS = int(os.environ.get("SNN_MLP_STEPS", "30000"))
LR = float(os.environ.get("SNN_MLP_LR", "5e-6"))
LR_MIN = LR / 10
WARMUP = int(os.environ.get("SNN_MLP_WARMUP", "500"))
CLIP = 1.0
AMP = os.environ.get("SNN_MLP_AMP", "1") == "1"
RESUME = os.environ.get("SNN_MLP_RESUME", os.path.join(W, "SNN34_distill", "snn34_dg7.best.pt"))
BATCH = int(os.environ.get("SNN_MLP_BATCH", "128"))
WIN = 128
DATA = r"C:\wanjuan_local"
EXTRA = os.environ.get("SNN_MLP_EXTRA", os.path.join(DATA, "wanjuan_extra_bpe.bin"))

OUTDIR = os.path.join(W, "SNN34_distill")
TAG = os.environ.get("SNN_MLP_TAG", "mlp")
BEST = os.path.join(OUTDIR, "snn34_%s.best.pt" % TAG)
LATEST = os.path.join(OUTDIR, "snn34_%s.pt" % TAG)
LOG = os.path.join(OUTDIR, "_%s_launch.log" % TAG)

DIM = int(os.environ.get("SNN_MLP_DIM", "640"))
NLAY = int(os.environ.get("SNN_MLP_LAYERS", "10"))
NHED = int(os.environ.get("SNN_MLP_HEADS", "8"))
DEV = "cuda"


def log(m):
    print(m, flush=True)
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        f.write("%s %s\n" % (datetime.datetime.now().strftime("%m-%d %H:%M:%S"), m))


def window_ok(now=None):
    now = now or datetime.datetime.now()
    if now.weekday() < 5:
        return not (18.0 <= now.hour + now.minute / 60.0 < 20.0)
    return True


class MlpBlock(SpikingBlock):
    """同构 block，唯一差异：FFN 中段 lif2 脉冲 → ReLU。注意力 lif1 门控保留。"""

    def forward(self, x):
        a = self.attn(self.norm1(x))
        s = self.lif1(a)
        x = x + a * (0.5 + 0.5 * s)
        h = self.fc2(F.relu(self.fc1(self.norm2(x))))
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
POOLS = [train_tok]
if EXTRA and os.path.exists(EXTRA):
    ex = np.memmap(EXTRA, dtype=np.int32, mode="r")
    POOLS.append(ex)
    log("[data] extra pool %s = %d tok" % (EXTRA, len(ex)))
val_tok = np.load(os.path.join(DATA, "clean_val_bpe.npy"), mmap_mode="r")
WCOUNTS = [len(p) // WIN for p in POOLS]
PCUM = np.cumsum([c / sum(WCOUNTS) for c in WCOUNTS])
log("[data] pools=%s (%d tok) | val=%d tok | vocab=%d"
    % (["%d" % len(p) for p in POOLS], sum(len(p) for p in POOLS), len(val_tok), VOCAB))

# 三线探针（种子 2026，与 dg7 探针完全一致）
PROBES = {}
if EXTRA and os.path.exists(EXTRA):
    _seg = len(ex) // 16
    _rng = np.random.default_rng(2026)
    for pname, (a, b) in {"law": (3, 5), "news": (2, 3), "gen": (6, 16)}.items():
        _lo, _hi = a * _seg, min(b * _seg, len(ex) - WIN)
        _starts = _rng.integers(_lo, _hi, size=32)
        PROBES[pname] = np.stack([np.asarray(ex[s:s + WIN], dtype=np.int64)
                                  for s in _starts])

model = MlpVariant(VOCAB, DIM, NLAY, NHED, WIN, alpha=1.0).to(DEV)
start_step = 0
if os.path.exists(LATEST) and os.environ.get("SNN_MLP_FRESH", "0") != "1":
    ck = torch.load(LATEST, map_location=DEV, weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if not k.endswith(".v")}
    model.load_state_dict(sd, strict=False)
    start_step = int(ck.get("meta", {}).get("step", 0))
    best = float(ck.get("meta", {}).get("val", 1e9))
    log("[model] resume %s @ step %d (val %.4f)" % (LATEST, start_step, best))
elif RESUME and os.path.exists(RESUME):
    ck = torch.load(RESUME, map_location=DEV, weights_only=False)
    sd = {k: v for k, v in ck["model"].items() if not k.endswith(".v")}
    model.load_state_dict(sd, strict=False)
    best = 1e9
    log("[model] warm-start %s (step=%s)" % (RESUME, ck.get("meta", {}).get("step")))
else:
    best = 1e9
    log("[model] random init")
log("[model] params=%.1fM | MLP-FFN variant" % (sum(p.numel() for p in model.parameters()) / 1e6))

opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01, betas=(0.9, 0.95))


def get_lr(step):
    if step <= WARMUP:
        return LR * step / WARMUP
    prog = (step - WARMUP) / max(1, STEPS - WARMUP)
    return LR_MIN + (LR - LR_MIN) * 0.5 * (1.0 + math.cos(math.pi * prog))


def save(path, step, val):
    torch.save({"config": {"vocab": VOCAB, "dim": DIM, "n_layers": NLAY,
                           "n_heads": NHED, "block_size": WIN, "ff_mult": 4,
                           "alpha": 1.0, "gated_attn": True, "mlp_ffn": True},
                "model": model.state_dict(),
                "meta": {"step": step, "val": val}}, path)


def sample_batch():
    u = np.random.random()
    p = POOLS[0] if u <= PCUM[0] else POOLS[1]
    i = np.random.randint(0, len(p) // WIN) * WIN
    return np.asarray(p[i:i + BATCH * WIN], dtype=np.int64).reshape(BATCH, WIN)


@torch.no_grad()
def eval_ce(x):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, x.shape[0], 16):
        b = torch.from_numpy(np.ascontiguousarray(x[i:i + 16])).to(DEV)
        logits = model(b)
        lo = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), b[:, 1:].reshape(-1))
        tot += lo.item() * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / n


vx = np.stack([np.asarray(val_tok[i * WIN:(i + 1) * WIN], dtype=np.int64) for i in range(32)])
best = eval_ce(vx)
log("[init] val=%.4f (warm-start base) | STEPS=%d LR=%g AMP=%s" % (best, STEPS, LR, AMP))
if not window_ok():
    log("[guard] forbidden window @ 0, sleep until window ends")
    while not window_ok():
        time.sleep(300)
    log("[guard] window reopened @ 0")

t0 = time.time()
for step in range(start_step + 1, STEPS + 1):
    lr = get_lr(step)
    for g in opt.param_groups:
        g["lr"] = lr
    x = torch.from_numpy(sample_batch()).to(DEV)
    if AMP:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            lo = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    else:
        logits = model(x)
        lo = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), x[:, 1:].reshape(-1))
    opt.zero_grad()
    lo.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step()

    if step % 100 == 0:
        done = step - start_step
        tps = done * BATCH * WIN / max(1.0, time.time() - t0)
        log("step %d/%d | loss=%.4f lr=%.2e gnorm=%.2f | %.0fs | %.0f tok/s%s"
            % (step, STEPS, lo.item(), lr, gn.item(), time.time() - t0, tps,
               " [bf16]" if AMP else ""))
    if step % 2000 == 0 or step == STEPS:
        v = eval_ce(vx)
        pr = {k: eval_ce(px) for k, px in PROBES.items()}
        log("[eval] step %d val=%.4f (law=%.3f news=%.3f gen=%.3f) (best %.4f)"
            % (step, v, pr["law"], pr["news"], pr["gen"], best))
        if v < best:
            best = v
            save(BEST, step, v)
            log("[best] saved @ %d val=%.4f" % (step, v))
        save(LATEST, step, v)
    if step % 10000 == 0:
        time.sleep(2)   # 阶段性让出 I/O

log("[DONE] final best=%.4f" % best)
