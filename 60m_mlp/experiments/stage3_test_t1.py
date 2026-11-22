# -*- coding: utf-8 -*-
"""T1 少样本适应测试 v2（函数式，含 eta 扫描）。

协议：
  1) 领域 A 预热（BPTT）-> wrap
  2) 对照：前向无塑性（应与 zero-shot 一致）
  3) eta 扫描：调质门控 ΔW 局部适应，找有效区间
  4) 全参微调上限参考（最后跑，避免污染起点）
"""
import math, sys, os
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snn_minilm import SpikingMiniLM
from stage3_modulator import (PlasticityConfig, PlasticLinear,
                              stage3_wrap)

torch.manual_seed(7)
DEV = 'cpu'
V, DIM, LAYERS = 512, 64, 2
GDATA = torch.Generator().manual_seed(1234)

def make_seq(domain, T=64):
    base = [(2*i) % V for i in range(T)] if domain == 'A' \
        else [(1 + 2*i) % (V - 2) + 2 for i in range(T)]
    noise = (torch.rand(T, generator=GDATA) < 0.2).nonzero().flatten()
    for i in noise:
        base[i] = (base[i] + torch.randint(1, 5, (1,), generator=GDATA).item()) % V
    return torch.tensor(base)

def batch(ds, T=64):
    return torch.stack([make_seq(d, T) for d in ds])

def ce(model, idx):
    model.reset_states()
    logits = model(idx[:, :-1])
    return F.cross_entropy(logits.reshape(-1, V), idx[:, 1:].reshape(-1)).item()

def main():
    # ---- 1) 预热 ----
    model = SpikingMiniLM(V, dim=DIM, n_layers=LAYERS, n_heads=4, block_size=64).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    print('[pretrain] domain A ...')
    for step in range(200):
        idx = batch(['A'] * 8); model.reset_states()
        lg = model(idx[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, V), idx[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    base_ce = ce(model, batch(['B'] * 8))
    print('zero-shot CE on B(批次1): %.4f' % base_ce)

    adapt_seqs = [batch(['B'] * 4) for _ in range(4)]
    test_idx = batch(['B'] * 8)

    cfg = PlasticityConfig()
    model = stage3_wrap(model, cfg)

    def set_plastic(on):
        model.plastic_on = on
        for b in model.blocks: b.plastic_on = on

    def reset_all_dw():
        for b in model.blocks:
            for m in b.modules():
                if isinstance(m, PlasticLinear): m.reset_plastic()

    def trial(eta):
        for b in model.blocks:
            for m in b.modules():
                if isinstance(m, PlasticLinear):
                    m.cfg.eta = eta; m.reset_plastic()
        set_plastic(False)
        with torch.no_grad():
            for s in adapt_seqs:
                model.reset_states()
                sup = ce(model, s)                     # 关塑性测 surprise
                set_plastic(True)
                for b in model.blocks:
                    b.m_const = b.novelty_m(sup, cfg)  # 新颖性调质
                model.reset_states(); model(s)         # 塑性适应
                set_plastic(False)
        model.reset_states()
        return ce(model, test_idx)

    # ---- 2) 对照 ----
    ctrl = trial(0.0)          # eta=0 即纯前向对照（同一协议零塑性）
    print('对照(eta=0, 无塑性写入): %.4f' % ctrl)

    # ---- 3) eta 扫描 ----
    print('--- eta 扫描（调质门控 ΔW）---')
    results = {}
    for eta in (0.0002, 0.0005, 0.001, 0.002, 0.005):
        r = trial(eta)
        results[eta] = r
        print('  eta=%.4f: %.4f %s' % (eta, r, 'OK' if r < ctrl else 'X'))
    best_eta = min(results, key=results.get)
    ok = results[best_eta] < ctrl
    print('best eta=%.4f (%.4f) %s' % (best_eta, results[best_eta],
          '塑性有效' if ok else '全区间无效'))

    # ---- 4) 全参微调上限参考 ----
    reset_all_dw(); set_plastic(False)
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(8):
        idx = adapt_seqs[step % len(adapt_seqs)]; model.reset_states()
        lg = model(idx[:, :-1])
        loss = F.cross_entropy(lg.reshape(-1, V), idx[:, 1:].reshape(-1))
        opt2.zero_grad(); loss.backward(); opt2.step()
    ft = ce(model, test_idx)
    print('全参微调上限: %.4f' % ft)
    print('=' * 56)
    print('zero-shot %.4f | ctrl %.4f | plastic(%.4f) %.4f | finetune %.4f'
          % (base_ce, ctrl, best_eta, results[best_eta], ft))

if __name__ == '__main__':
    main()
