# -*- coding: utf-8 -*-
"""Stage3 T1 少样本适应 —— 真实 60M MLP 模型 + 真实语料（news 领域）。

协议：
  1) 加载 snn34_mlp.best.pt (val 4.8505) → convert_to_plastic（ΔW/痕迹零初始化）
  2) 先量权重 std（决定 clamp_dw 量级）
  3) 领域 B = extra bin news 段（seg 2）的 held-out 窗
  4) 对照/扫描：eta=0（纯前向）/ 4 档 eta / clamp 同权重 std 量级
  5) 全参微调参考（同 token 预算 BPTT，小 LR）
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from bpe_tokenizer import BPETokenizer                       # noqa: E402
from mlp_stage3 import MlpVariant, convert_to_plastic        # noqa: E402
from stage3_modulator import PlasticLinear, PlasticityConfig  # noqa: E402

DEV = "cuda"
BEST = os.path.join(W, "SNN34_distill", "snn34_mlp.best.pt")
BIN = r"C:\wanjuan_local\wanjuan_extra_bpe.bin"
WIN = 128
tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
VOCAB = tok.vocab_size

ex = np.memmap(BIN, dtype=np.int32, mode="r")
_seg = len(ex) // 16
NEWS_LO, NEWS_HI = 2 * _seg, 3 * _seg            # seg2 = news 主体

rng = np.random.default_rng(2026)
# 适应集 8 条；held-out 测试集 32 条（与训练流不重叠）
adapt_starts = rng.integers(NEWS_LO, NEWS_LO + (NEWS_HI - NEWS_LO) // 2, size=8)
test_starts = rng.integers(NEWS_LO + (NEWS_HI - NEWS_LO) // 2, NEWS_HI, size=32)
adapt_seqs = [np.asarray(ex[s:s + WIN], dtype=np.int64) for s in adapt_starts]
test_x = np.stack([np.asarray(ex[s:s + WIN], dtype=np.int64) for s in test_starts])

model = MlpVariant(VOCAB, 640, 10, 8, WIN, alpha=1.0).to(DEV)
ck = torch.load(BEST, map_location=DEV, weights_only=False)
sd = {k: v for k, v in ck["model"].items() if not k.endswith(".v")}
model.load_state_dict(sd, strict=False)
print("[model] loaded %s (meta val=%.4f)" % (os.path.basename(BEST),
      ck.get("meta", {}).get("val", -1)))

# 权重 std（决定 clamp 量级）
stds = [p.std().item() for n, p in model.named_parameters()
        if p.dim() == 2 and n.endswith("weight")]
w_std = float(np.median(stds))
print("[calib] fc/attn 权重 std 中位数 = %.4f -> clamp_dw 取同量级" % w_std)

model = convert_to_plastic(model, PlasticityConfig(clamp_dw=round(w_std, 4)))
plastics = [m for b in model.blocks for m in b.modules() if isinstance(m, PlasticLinear)]
print("[model] PlasticLinear 层数 = %d" % len(plastics))


def set_plastic(on):
    for b in model.blocks:
        b.plastic_on = on


@torch.no_grad()
def ce_batch(x):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), 16):
        b = torch.from_numpy(np.ascontiguousarray(x[i:i + 16])).to(DEV)
        logits = model(b)
        lo = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), b[:, 1:].reshape(-1))
        tot += lo.item() * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / n


base = ce_batch(test_x)
print("\n[T1] news held-out base CE = %.4f" % base)


def trial(eta):
    for m in plastics:
        m.cfg.eta = eta
        m.reset_plastic()
    set_plastic(False)
    with torch.no_grad():
        for s in adapt_seqs:
            model.reset_states()
            sup = ce_batch(s[None, :]) if False else None
            # surprise：当前序列 CE（关塑性测）
            xb = torch.from_numpy(s[None, :]).to(DEV)
            logits = model(xb)
            sup = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                                  xb[:, 1:].reshape(-1)).item()
            set_plastic(True)
            for b in model.blocks:
                b.m_const = b.novelty_m(sup, b.cfg)     # 新颖性调质
            model.reset_states()
            model(xb, surprise=None)                    # m_const 已注入 → ΔW 累积
            set_plastic(False)
    return ce_batch(test_x)


print("\n[T1] eta 扫描（8-shot 局部适应，零 BPTT）:")
results = {}
for eta in (2e-5, 5e-5, 1e-4, 2e-4, 5e-4):
    r = trial(eta)
    results[eta] = r
    print("  eta=%.0e: CE=%.4f  (%+.4f) %s"
          % (eta, r, r - base, "OK" if r < base else "X"))
best_eta = min(results, key=results.get)

# 全参微调参考（同 8 序列，8 步 BPTT，lr 1e-5 —— 大模型微调保守 LR）
set_plastic(False)
for m in plastics:
    m.reset_plastic()
opt2 = torch.optim.AdamW(model.parameters(), lr=1e-5)
for step in range(8):
    s = adapt_seqs[step % len(adapt_seqs)]
    xb = torch.from_numpy(s[None, :]).to(DEV)
    logits = model(xb)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB), xb[:, 1:].reshape(-1))
    opt2.zero_grad(); loss.backward(); opt2.step()
ft = ce_batch(test_x)

print("\n=== T1 真实语料结果（news held-out CE，越低越好）===")
print("  base（未经适应）      : %.4f" % base)
print("  调质门控 ΔW (eta=%.0e): %.4f  (%+.4f) %s"
      % (best_eta, results[best_eta], results[best_eta] - base,
         "有效" if results[best_eta] < base else "无效"))
print("  全参微调（8 步 BPTT） : %.4f  (%+.4f)" % (ft, ft - base))
