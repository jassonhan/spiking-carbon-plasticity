# -*- coding: utf-8 -*-
"""Stage3 反馈对齐（Feedback Alignment）——带方向性的局部学习规则。

与 Hebbian 三因子的根本区别：
  Hebbian:  ΔW ∝ pre ⊗ post            （无方向，只强化共现 → 天花板 -0.005）
  FA:       ΔW ∝ -δ ⊗ a，δ = B·δ_next  （δ 携带真实预测误差方向）

跨层用**固定随机反馈矩阵** B（不训练）传误差；本层内用 W^T（局部可用）。
生物对应：皮层反馈连接（随机、固定）+ 局部突触更新。

结构：
  FAMlpVariant(MlpVariant)
    - 每 block 一个固定 B_block (dim×dim)，N(0, 1/dim) 初始化
    - fa_adapt(idx): 前向记录 → 输出层误差 → 逐 block 反向（no_grad）→ 更新 fc1/fc2 的 dw
    - dw 与 PlasticLinear 共用（前向恒用 W+dw；fw 仍由 BPTT 训练，dw 只是快权重）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_minilm import SpikingBlock, SpikingMiniLM
from stage3_modulator import PlasticLinear


class MlpBlock(SpikingBlock):
    def forward(self, x):
        a = self.attn(self.norm1(x))
        s = self.lif1(a)
        x = x + a * (0.5 + 0.5 * s)
        h = self.fc2(F.relu(self.fc1(self.norm2(x))))
        return x + h


class FAMlpBlock(MlpBlock):
    """带固定随机反馈矩阵的块（fc1/fc2 → PlasticLinear，dw 由 FA 更新）。"""

    def __init__(self, dim, n_heads, block_size, ff_mult=4, alpha=1.0,
                 gated_attn=True, fb_scale=None):
        super().__init__(dim, n_heads, block_size, ff_mult, alpha=alpha,
                         gated_attn=gated_attn)
        self.fc1 = PlasticLinear(self.fc1, _DwCfg())
        self.fc2 = PlasticLinear(self.fc2, _DwCfg())
        scale = fb_scale if fb_scale is not None else 1.0 / math.sqrt(dim)
        # 固定随机反馈矩阵（跨 block，dim → dim）
        self.register_buffer("B", torch.randn(dim, dim) * scale)

    def record(self, x):
        """前向并返回本层中间量（供 FA 反向）。"""
        a = self.attn(self.norm1(x))
        s = self.lif1(a)
        x2 = x + a * (0.5 + 0.5 * s)
        pre1 = self.norm2(x2)
        z = F.linear(pre1, self.fc1.base.weight + self.fc1.dw, self.fc1.base.bias)
        act = F.relu(z)
        h = F.linear(act, self.fc2.base.weight + self.fc2.dw, self.fc2.base.bias)
        return x2 + h, {"pre1": pre1, "z": z, "act": act}


class _DwCfg:
    """PlasticLinear 需要的最小配置（FA 不用 eta/delta/gamma，只留 dw 存储）。"""

    def __init__(self):
        self.eta = 0.0
        self.delta = 0.85
        self.gamma = 0.0
        self.clamp_dw = 0.02
        self.tau = 1.0
        self.m_max = 1.0
        self.wrap_ffn = True


class FAMlpVariant(MlpVariant if False else SpikingMiniLM):
    """FA 版模型：block 自带反馈矩阵，fa_adapt() 做局部有方向更新。"""

    def __init__(self, vocab, dim, n_layers, n_heads, block_size, alpha=1.0):
        super().__init__(vocab, dim=dim, n_layers=n_layers, n_heads=n_heads,
                         block_size=block_size, alpha=alpha, gated_attn=True)
        self.blocks = nn.ModuleList(
            [FAMlpBlock(dim, n_heads, block_size, 4, alpha=alpha, gated_attn=True)
             for _ in range(n_layers)])

    def forward(self, idx):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))

    @torch.no_grad()
    def fa_adapt(self, idx, lr, clamp=0.02, head_lr_scale=1.0, decay=0.0):
        """一遍 FA：前向记录 → 输出误差 → 逐 block 局部更新 dw（不碰 base 权重）。

        idx: (B,T)；loss = CE(logits[:, :-1], idx[:, 1:])
        decay: dw 每步衰减（0.01-0.05 可让多遍稳定，防饱和过拟合）
        返回该序列的 CE 值。
        """
        B, T = idx.shape
        # ---- 前向记录 ----
        x = self.tok_emb(idx)
        recs = []
        for blk in self.blocks:
            x, rec = blk.record(x)
            recs.append(rec)
        logits = self.lm_head(self.ln_f(x))
        # ---- 输出层误差（最后一层用真实梯度：lm_head 是最后一层，局部可得其转置） ----
        lg = logits[:, :-1]                                # (B,T-1,V)
        tgt = idx[:, 1:]                                   # (B,T-1)
        ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1)).item()
        p = F.softmax(lg.reshape(-1, lg.shape[-1]), dim=-1)
        onehot = F.one_hot(tgt.reshape(-1), lg.shape[-1]).float()
        d_logits = (p - onehot) / (B * (T - 1))            # (B*(T-1), V)
        # 穿过 lm_head + ln_f（真实转置，最后一层）
        d_x = (d_logits @ self.lm_head.weight).reshape(B, T - 1, -1)
        # 补回 seq 维（最后一个位置无误差）
        d_x = F.pad(d_x, (0, 0, 0, 1))                     # (B,T,D)
        # ---- 逐 block 反向（跨层固定随机 B） ----
        for li in range(len(self.blocks) - 1, -1, -1):
            blk = self.blocks[li]
            rec = recs[li]
            d_out = d_x                                    # (B,T,D) 本 block 输出误差
            # 残差直通：h 的梯度 = d_out
            d_h = d_out
            # fc2: ΔW = -lr * δ_h^T · act
            dW2 = torch.einsum("bth,bto->ho", d_h, rec["act"]) * (-lr)
            if decay > 0:
                blk.fc2.dw.mul_(1.0 - decay)
            blk.fc2.dw.add_(dW2).clamp_(-clamp, clamp)
            # act 梯度：经 W_fc2 转置（本层）
            d_act = d_h @ (blk.fc2.base.weight + blk.fc2.dw)
            d_z = d_act * (rec["z"] > 0).float()
            dW1 = torch.einsum("bth,bti->hi", d_z, rec["pre1"]) * (-lr)
            if decay > 0:
                blk.fc1.dw.mul_(1.0 - decay)
            blk.fc1.dw.add_(dW1).clamp_(-clamp, clamp)
            # 传到上一层：固定随机 B
            d_x = d_out @ blk.B.T
        return ce

    def reset_dw(self):
        for b in self.blocks:
            b.fc1.dw.zero_(); b.fc1.e.zero_()
            b.fc2.dw.zero_(); b.fc2.e.zero_()


def load_fa_from(path, vocab, dim=640, n_layers=10, n_heads=8, block_size=128,
                 dev="cuda"):
    """从 MLP 主线 checkpoint 构造 FA 模型（base 权重全保留，dw=0）。"""
    m = FAMlpVariant(vocab, dim, n_layers, n_heads, block_size, alpha=1.0).to(dev)
    ck = torch.load(path, map_location=dev, weights_only=False)
    sd = {}
    for k, v in ck["model"].items():
        if k.endswith(".v"):
            continue
        if k.startswith(("fc1.", "fc2.")):
            continue                        # 单独处理（fc1.weight → fc1.base.weight）
        sd[k] = v
    missing, unexpected = m.load_state_dict(sd, strict=False)
    # fc1/fc2 权重映射
    with torch.no_grad():
        for i, blk in enumerate(m.blocks):
            for p in ("fc1", "fc2"):
                w = ck["model"]["blocks.%d.%s.weight" % (i, p)]
                getattr(blk, p).base.weight.copy_(w)
    return m


if __name__ == "__main__":
    # 冒烟：小模型验证 FA 能降低 loss（合成序列）
    torch.manual_seed(7)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    V, D, L = 512, 64, 2
    m = FAMlpVariant(V, D, L, 4, 128).to(dev)
    print("[smoke] params=%.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))

    g = torch.Generator().manual_seed(1)
    # 合成任务：固定循环模式（可学）
    base = torch.randint(0, 50, (128,), generator=g)
    seqs = [base.clone() for _ in range(8)]

    def eval_ce():
        m.eval(); tot = 0
        with torch.no_grad():
            for s in seqs:
                xb = s[None, :].to(dev)
                lg = m(xb[:, :-1])
                tot += F.cross_entropy(lg.reshape(-1, V), xb[:, 1:].reshape(-1)).item()
        m.train()
        return tot / len(seqs)

    ce0 = eval_ce()
    print("[smoke] 初始 CE = %.3f" % ce0)
    for ep in range(30):
        for s in seqs:
            m.fa_adapt(s[None, :].to(dev), lr=2e-3, clamp=0.02)
        if (ep + 1) % 10 == 0:
            print("[smoke] ep %2d: CE = %.3f" % (ep + 1, eval_ce()))
