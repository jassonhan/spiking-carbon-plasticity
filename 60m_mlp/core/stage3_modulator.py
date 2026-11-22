# -*- coding: utf-8 -*-
"""
Stage3 调质门控可塑性原型（v1.1）—— 三因子局部学习规则
============================================================
v1.0 教训（T1 测试证伪）：
  1) 同时刻 Hebbian (post_t ⊗ pre_t) 学不到时序转移 → 引入资格痕迹 e：
     e = delta*e + outer(post_t, pre_t)   # 痕迹桥接因果滞后
     ΔW += eta * M * e                     # tag-and-capture
  2) M=exp(-surprise) 语义反了（高错误=最需要学习时反而关门）
     → 新颖性驱动：M = min(surprise / tau, m_max)   # 错误越大写入越强

生物对应：
  e = 突触"标签"(eligibility trace)，桥接因果滞后直到调质到达
  M = 新颖性/乙酰胆碱样信号（预测误差大 → 可塑窗口开）
  ΔW = 快权重，局部规则更新，无 BPTT

慢权重 W 仍走 BPTT（"先教后放"的教）；ΔW 为"放"的局部组件。
SpikingBlock 子类化，参数名兼容，可 warm-start，不改主线文件。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_minilm import (SpikeFn, LifNeuron, CausalSelfAttn,
                        SpikingBlock, SpikingMiniLM)


class PlasticityConfig:
    def __init__(self,
                 eta=0.0002,         # 写入率（T1 标定最优；>0.005 过度可塑反噬）
                 delta=0.85,         # 资格痕迹衰减（桥接 ~1/(1-delta) 步因果滞后）
                 gamma=0.002,        # ΔW 衰减
                 tau=4.0,            # 新颖性标尺（CE 单位）
                 m_max=1.5,          # 调质上限
                 wrap_ffn=True,
                 clamp_dw=0.02):     # ΔW 钳位（必须 ≤ W 的 std 量级，否则反噬）
        self.eta = eta; self.delta = delta; self.gamma = gamma
        self.tau = tau; self.m_max = m_max
        self.wrap_ffn = wrap_ffn; self.clamp_dw = clamp_dw


class PlasticLinear(nn.Module):
    """包裹已有 nn.Linear：资格痕迹 e + 快权重 ΔW（均不进 BPTT 图）。"""

    def __init__(self, base: nn.Linear, cfg: PlasticityConfig):
        super().__init__()
        self.base = base
        self.cfg = cfg
        out_f, in_f = base.weight.shape
        self.register_buffer("dw", torch.zeros(out_f, in_f))
        self.register_buffer("e", torch.zeros(out_f, in_f))

    @torch.no_grad()
    def update_token(self, pre_t, post_t, m):
        """单 token 局部更新。pre_t:(...,in_f) post_t:(...,out_f)（batch 维取均值）。m:标量。"""
        p = pre_t.reshape(-1, pre_t.shape[-1])       # (N, in_f)
        q = post_t.reshape(-1, post_t.shape[-1])     # (N, out_f)
        self.e.mul_(self.cfg.delta).add_(q.T @ p / p.shape[0])
        self.dw.mul_(1.0 - self.cfg.gamma).add_(self.cfg.eta * m * self.e)
        self.dw.clamp_(-self.cfg.clamp_dw, self.cfg.clamp_dw)

    @torch.no_grad()
    def reset_plastic(self):
        self.dw.zero_(); self.e.zero_()

    def forward(self, x):
        return F.linear(x, self.base.weight + self.dw, self.base.bias)


class Stage3SpikingBlock(SpikingBlock):
    """带调质门控可塑性的脉冲块（参数名兼容，可 warm-start）。"""

    def __init__(self, dim, n_heads, block_size, ff_mult=4, beta=0.9, thresh=1.0,
                 alpha=2.0, gated_attn=False, cfg: PlasticityConfig = None):
        super().__init__(dim, n_heads, block_size, ff_mult, beta, thresh, alpha,
                         gated_attn)
        self.cfg = cfg or PlasticityConfig()
        self.plastic_on = True
        self.m_const = None              # 外部注入口（每序列）
        if self.cfg.wrap_ffn:
            self.fc1 = PlasticLinear(self.fc1, self.cfg)
            self.fc2 = PlasticLinear(self.fc2, self.cfg)

    @staticmethod
    def novelty_m(surprise, cfg):
        """新颖性调质：预测误差越大，可塑窗口越开。"""
        return min(abs(float(surprise)) / cfg.tau, cfg.m_max)

    def forward(self, x, surprise=None):
        B, T, D = x.shape
        a_all = self.attn(self.norm1(x))
        if self.gated_attn:
            s_all = self.lif1(a_all)
            x = x + a_all * (0.5 + 0.5 * s_all)
        else:
            x = x + self.lif1(a_all)

        fc1, fc2 = self.fc1, self.fc2
        use_plastic = self.plastic_on and isinstance(fc1, PlasticLinear)
        if use_plastic:
            m = self.m_const if self.m_const is not None \
                else self.novelty_m(surprise or 0.0, self.cfg)
        else:
            m = None

        pre1 = self.norm2(x)
        z = fc1(pre1)                              # (B,T,H) 慢权重+ΔW 前向
        H = fc1.base.weight.shape[0]
        spikes_out = torch.zeros(B, T, H, device=x.device, dtype=x.dtype)
        lif2 = self.lif2
        if lif2.v.numel() != B * H:
            lif2.reset(B, H, x.device, x.dtype)
        v = lif2.v
        for t in range(T):
            v = lif2.beta * v + z[:, t, :]
            s = SpikeFn.apply(v, lif2.alpha, lif2.thresh)
            v = v - s.detach() * lif2.thresh
            spikes_out[:, t, :] = s
            if use_plastic:
                fc1.update_token(pre1[:, t, :], s, m)      # 因果配对：pre_t vs post_t
        lif2.v = v.detach()
        h = fc2(spikes_out)
        if use_plastic:
            for t in range(T):
                fc2.update_token(spikes_out[:, t, :], h[:, t, :], m)
        x = x + h
        return x


def stage3_wrap(model: SpikingMiniLM, cfg: PlasticityConfig) -> SpikingMiniLM:
    """把 SpikingMiniLM 的 blocks 原位替换为 Stage3 版（参数继承，可 warm-start）。"""
    for i, blk in enumerate(model.blocks):
        nb = Stage3SpikingBlock(blk.attn.dim, blk.attn.n_heads, model.block_size,
                                cfg=cfg, gated_attn=blk.gated_attn)
        sd = {}
        for k, v in blk.state_dict().items():
            if k.endswith('.v'):
                continue
            for pref in ('fc1.', 'fc2.'):
                if k.startswith(pref):
                    k = pref + 'base.' + k[len(pref):]
            sd[k] = v
        nb.load_state_dict(sd, strict=False)
        model.blocks[i] = nb
    return model
