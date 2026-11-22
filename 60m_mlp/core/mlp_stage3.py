# -*- coding: utf-8 -*-
"""Stage3 调质门控 ΔW —— MLP 主线接入版。

消融实验（2026-11-19）证明 LIF FFN 是质量+速度瓶颈，主线转 MLP 后，
本模块把 Stage3 的三因子局部学习规则（资格痕迹 e × 新颖性调质 M → ΔW）
接到 MLP 变体的 fc1/fc2 上。复用 stage3_modulator.py 的 PlasticLinear 与
T1 标定参数（eta=2e-4, clamp_dw=0.02），参数名兼容可原地转换。

协议（先教后放）：
  - 训练期：plastic_on=False，dw 冻结为 0，forward 等价普通 MLP（BPTT 主干"教"）
  - 适应期：plastic_on=True，逐 token 局部更新 dw（ novelty_m(surprise) 门控）
  - 测试期：plastic_on=False（dw 保留生效——前向恒用 W+dw）
"""
import torch
import torch.nn.functional as F

from snn_minilm import SpikingBlock, SpikingMiniLM
from stage3_modulator import PlasticityConfig, PlasticLinear


class MlpBlock(SpikingBlock):
    """与 snn_pretrain_mlp.py 同款（自包含副本，避免 import 训练脚本副作用）。"""

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


def build_cfg(eta, clamp_dw, mod_mode="token_ce", tau_token=2.0, m_floor=0.05):
    """构造兼容 PlasticLinear（读 eta/delta/gamma/clamp_dw）的配置。"""
    cfg = PlasticityConfig(eta=eta, clamp_dw=clamp_dw)
    cfg.mod_mode = mod_mode
    cfg.tau_token = tau_token
    cfg.m_floor = m_floor
    return cfg


def token_modulation(logits, idx, tau_token=2.0, m_floor=0.1, m_max=1.5):
    """误差驱动三因子：token 级相对预测误差 → (B,T1) 调质。

    s_t = -log p(idx_t | idx_<t)（逐 token CE，局部可算）
    m_t = clip(s_t / s_bar, m_floor, m_max)，s_bar = 该序列平均 CE
      → 相对调制：预测失败的 token m>1（多学），已会的 token m<1（少学）
      → 自适应归一化，不依赖绝对 CE 尺度（避免全饱和）
    """
    import torch.nn.functional as F
    B, T1, V = logits.shape
    tgt = idx[:, 1:1 + T1]                     # (B,T1) 下一个 token
    s = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                        reduction="none").reshape(B, T1)
    s_bar = s.mean(dim=1, keepdim=True)
    m = (1.0 + (s - s_bar) / max(tau_token, 1e-6)).clamp(m_floor, m_max)
    return m


class MlpPlasticBlock(MlpBlock):
    """带调质门控可塑性的 MLP 块（参数名兼容，可从 MlpBlock 原地转换）。"""

    def __init__(self, dim, n_heads, block_size, ff_mult=4, alpha=1.0,
                 gated_attn=True, cfg: PlasticityConfig = None):
        super().__init__(dim, n_heads, block_size, ff_mult, alpha=alpha,
                         gated_attn=gated_attn)
        self.cfg = cfg or PlasticityConfig()
        self.plastic_on = True
        self.m_const = None                     # 序列级外部注入
        self.token_mod = None                   # token 级外部注入 (B,T)
        if self.cfg.wrap_ffn:
            self.fc1 = PlasticLinear(self.fc1, self.cfg)
            self.fc2 = PlasticLinear(self.fc2, self.cfg)

    @staticmethod
    def novelty_m(surprise, cfg):
        """新颖性调质：预测误差越大，可塑窗口越开。"""
        return min(abs(float(surprise)) / cfg.tau, cfg.m_max)

    def forward(self, x, surprise=None, token_mod=None):
        a = self.attn(self.norm1(x))
        s = self.lif1(a)
        x = x + a * (0.5 + 0.5 * s)

        fc1, fc2 = self.fc1, self.fc2
        use_plastic = self.plastic_on and isinstance(fc1, PlasticLinear)
        tm = token_mod if token_mod is not None else self.token_mod
        if use_plastic:
            if tm is not None and getattr(self.cfg, "mod_mode", "novelty") == "token_ce":
                m_series = None                  # 逐 token 调质路径
            else:
                tm = None
                m_series = self.m_const if self.m_const is not None \
                    else self.novelty_m(surprise or 0.0, self.cfg)
        else:
            tm = None
            m_series = None

        pre1 = self.norm2(x)
        z = fc1(pre1)                            # (B,T,H) 前向恒用 W+dw
        act = F.relu(z)
        T = x.shape[1]
        # fc1 的 token 级 pre 缓存（供 token_ce 调制对齐）
        fc1._last_pre = pre1.detach()
        if use_plastic:
            for t in range(T):
                m_t = float(tm[:, t].mean()) if tm is not None else m_series
                fc1.update_token(pre1[:, t, :], act[:, t, :], m_t)   # 因果配对
        h = fc2(act)
        fc2._last_pre = act.detach()
        if use_plastic:
            for t in range(T):
                m_t = float(tm[:, t].mean()) if tm is not None else m_series
                fc2.update_token(act[:, t, :], h[:, t, :], m_t)
        return x + h


def convert_to_plastic(model: MlpVariant, cfg: PlasticityConfig = None) -> MlpVariant:
    """原地转换：MlpVariant 各 block → MlpPlasticBlock（权重全保留）。

    参数名变化：fc1.weight → fc1.base.weight（dw/e 初始化零，T3/T1 从零开始）。
    """
    cfg = cfg or PlasticityConfig()
    dev = next(model.parameters()).device
    old_blocks = model.blocks
    new_blocks = torch.nn.ModuleList()
    for blk in old_blocks:
        nb = MlpPlasticBlock(blk.attn.dim, blk.attn.n_heads, model.block_size,
                             4, alpha=blk.lif1.alpha,
                             gated_attn=blk.gated_attn, cfg=cfg)
        sd = {k: v for k, v in blk.state_dict().items() if not k.endswith(".v")}
        for pref in ("fc1.", "fc2."):
            sd = {((pref + "base." + k[len(pref):]) if k.startswith(pref) else k): v
                  for k, v in sd.items()}
        missing, unexpected = nb.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert all(k.endswith((".v", ".dw", ".e")) for k in missing), \
            f"unexpected missing: {[k for k in missing if not k.endswith(('.v', '.dw', '.e'))][:5]}"
        nb.plastic_on = False                    # 默认训练模式（dw 冻结）
        nb.to(dev)                               # 跟随原模型设备（dw/e buffer 同步搬运）
        new_blocks.append(nb)
    model.blocks = new_blocks
    model.__class__ = PlasticMlpVariant         # 升级 forward 支持 surprise
    return model


class PlasticMlpVariant(MlpVariant):
    """顶层支持 surprise / token_mod → 各 block 调质信号注入。"""

    def forward(self, idx, surprise=None, token_mod=None):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x, surprise=surprise, token_mod=token_mod)
        x = self.ln_f(x)
        return self.lm_head(x)


if __name__ == "__main__":
    # 冒烟自测：转换等价性（plastic_on=False 时 forward 与原 MLP 数值一致）
    torch.manual_seed(7)
    m = MlpVariant(512, 64, 2, 4, 64, alpha=1.0)
    m.eval()
    idx = torch.randint(0, 512, (4, 32))
    with torch.no_grad():
        out_before = m(idx).clone()
    m = convert_to_plastic(m)
    m.eval()
    with torch.no_grad():
        out_after = m(idx).clone()
    diff = (out_before - out_after).abs().max().item()
    print("[smoke] convert 等价性 max|diff| = %.3e %s"
          % (diff, "OK" if diff < 1e-5 else "FAIL"))
    # 塑性开启路径冒烟（小前向不炸即可）
    m.blocks[0].plastic_on = True
    m.blocks[0].m_const = 0.5
    with torch.no_grad():
        _ = m(idx)
    print("[smoke] plastic forward OK, dw nonzero=%s"
          % any(p.abs().sum().item() > 0 for b in m.blocks
                for mod in b.modules() if isinstance(mod, PlasticLinear)
                for p in [mod.dw]))
