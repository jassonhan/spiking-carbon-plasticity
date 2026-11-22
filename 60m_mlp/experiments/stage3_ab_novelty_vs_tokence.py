# -*- coding: utf-8 -*-
"""Stage3 A/B：序列级 novelty vs token 级误差驱动（同一 60M 权重 + 英语新领域）。

指标：英文 held-out CE 改善 | 中文 val 漂移 | |dw|
低显存模式（小 batch），可与训练共存。
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

W = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, W)
from bpe_tokenizer import BPETokenizer                               # noqa: E402
from mlp_stage3 import (MlpVariant, convert_to_plastic, build_cfg,
                        token_modulation)                            # noqa: E402
from stage3_modulator import PlasticLinear                            # noqa: E402

DEV = "cuda"
WIN = 128
BEST = os.path.join(W, "SNN34_distill", "snn34_mlp.best.pt")
tok = BPETokenizer.load(os.path.join(W, "bpe_chinese_wanjuan.json"))
VOCAB = tok.vocab_size

en = np.load(r"E:\english_corpus\wiki_en_bpe.npy", mmap_mode="r")
zh = np.load(r"C:\wanjuan_local\clean_val_bpe.npy", mmap_mode="r")
rng = np.random.default_rng(2026)
adapt = [np.asarray(en[s:s + WIN], dtype=np.int64)
         for s in rng.integers(0, int(len(en) * 0.4), size=8)]
en_test = np.stack([np.asarray(en[s:s + WIN], dtype=np.int64)
                    for s in rng.integers(int(len(en) * 0.4), len(en) - WIN, size=16)])
zh_test = np.stack([np.asarray(zh[i * WIN:(i + 1) * WIN], dtype=np.int64)
                    for i in range(16)])


def load_model(mod_mode, eta, tau_token=2.0, m_floor=0.1):
    m = MlpVariant(VOCAB, 640, 10, 8, WIN, alpha=1.0).to(DEV)
    ck = torch.load(BEST, map_location=DEV, weights_only=False)
    m.load_state_dict({k: v for k, v in ck["model"].items() if not k.endswith(".v")},
                      strict=False)
    cfg = build_cfg(eta=eta, clamp_dw=0.002, mod_mode=mod_mode,
                    tau_token=tau_token, m_floor=m_floor)
    return convert_to_plastic(m, cfg)


@torch.no_grad()
def ce(model, x, bs=4):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(x), bs):
        b = torch.from_numpy(np.ascontiguousarray(x[i:i + bs])).to(DEV)
        lg = model(b)
        lo = F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB), b[:, 1:].reshape(-1))
        tot += lo.item() * b.shape[0]
        n += b.shape[0]
    model.train()
    return tot / n


def adapt_run(model, mode, reps=5):
    plastics = [m for b in model.blocks for m in b.modules()
                if isinstance(m, PlasticLinear)]
    for m in plastics:
        m.reset_plastic()
    model.eval()
    with torch.no_grad():
        for r in range(reps):
            for s in adapt:
                xb = torch.from_numpy(s[None, :]).to(DEV)
                if mode == "novelty":
                    for b in model.blocks:
                        b.plastic_on = False
                    lg = model(xb)
                    sup = F.cross_entropy(lg[:, :-1].reshape(-1, VOCAB),
                                          xb[:, 1:].reshape(-1)).item()
                    for b in model.blocks:
                        b.plastic_on = True
                        b.m_const = b.novelty_m(sup, b.cfg)
                    model(xb)
                else:   # token_ce
                    for b in model.blocks:
                        b.plastic_on = False
                    lg = model(xb[:, :-1])          # (1,127,V) 与 tgt=xb[:,1:] 对齐
                    tm = token_modulation(lg, xb, tau_token=model.blocks[0].cfg.tau_token,
                                          m_floor=model.blocks[0].cfg.m_floor)
                    for b in model.blocks:
                        b.plastic_on = True
                    model(xb[:, :-1], token_mod=tm)
                for b in model.blocks:
                    b.plastic_on = False
    dw = max(m.dw.abs().max().item() for m in plastics)
    return dw


print("加载基础模型测 base...", flush=True)
m0 = load_model("novelty", 5e-9)
en_base, zh_base = ce(m0, en_test), ce(m0, zh_test)
print("base: EN=%.4f ZH=%.4f\n" % (en_base, zh_base), flush=True)
del m0
torch.cuda.empty_cache()

rows = []
for mode in ("novelty", "token_ce"):
    for eta in (5e-9, 2e-8, 5e-8):
        m = load_model(mode, eta)
        dw = adapt_run(m, mode, reps=5)
        e, z = ce(m, en_test), ce(m, zh_test)
        rows.append((mode, eta, e, z, dw))
        print("%-8s eta=%.0e: EN=%.4f (%+.4f) ZH=%+.4f dw=%.5f"
              % (mode, eta, e, e - en_base, z - zh_base, dw), flush=True)
        del m
        torch.cuda.empty_cache()

print("\n=== A/B 汇总（EN 改善 / ZH 漂移）===")
for mode, eta, e, z, dw in rows:
    print("  %-8s eta=%.0e: %+.4f / %+.4f" % (mode, eta, e - en_base, z - zh_base))
