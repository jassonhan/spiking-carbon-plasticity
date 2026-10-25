"""
Spiking MiniLM —— 把 MiniLM 风格因果语言模型改造为脉冲神经网络(SNN, Spiking Neural Network)。

核心改动（相对 ANN 版 MiniLM）：
  - 用 LIF(Leaky Integrate-and-Fire, 漏积分触发)脉冲神经元替换 FFN / 注意力输出里的连续激活(GELU)；
  - 用替代梯度(surrogate gradient, arctan)绕过脉冲阈值不可导的问题做反向传播(BPTT 通过时间维)；
  - 把语言序列的「时间维」直接对齐到 SNN 的时序：token 逐个流入，膜电位跨 token 累积(因果递归)，
    与自回归 LM 天然契合(参考 SpikeGPT 的 "align sequence with temporal dimension" 思路)。

原型用途：验证「脉冲自注意力 + LIF + 替代梯度」管线能在中文语料上端到端训练、loss 能下降。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeFn(torch.autograd.Function):
    """Heaviside 前向(输出 0/1 脉冲)，arctan(alpha) 替代梯度反向。

    前向: spike = (v >= thresh)
    反向: d spike / d v ≈ d/dv [ (1/pi) * arctan(alpha*pi*(v-thresh)/2) ]
                   = (alpha/2) / (1 + (alpha*pi*(v-thresh)/2)^2)
    """

    @staticmethod
    def forward(ctx, v, alpha, thresh):
        ctx.alpha = alpha
        ctx.thresh = thresh
        ctx.save_for_backward(v)
        return (v >= thresh).float()

    @staticmethod
    def backward(ctx, grad_out):
        (v,) = ctx.saved_tensors
        u = ctx.alpha * math.pi * (v - ctx.thresh) / 2.0
        g = (ctx.alpha / 2.0) / (1.0 + u * u)
        return g * grad_out, None, None


class LifNeuron(nn.Module):
    """LIF 脉冲神经元。对输入序列 (B,T,D) 沿 T 维展开为递归动力学。

    v_t = beta * v_{t-1} + I_t          # 漏积分
    spike_t = (v_t >= thresh)            # 发放(Heaviside, 前向)
    v_t = v_t - spike_t * thresh         # 减性(软)重置
    膜电位 v 跨 token 携带，使 SNN 具备时序/因果递归特性。
    """

    def __init__(self, beta=0.9, thresh=1.0, alpha=2.0):
        super().__init__()
        self.beta = beta
        self.thresh = thresh
        self.alpha = alpha
        # (B,D) 膜电位；置为空，forward 时按当前 batch 自动初始化为 0
        self.register_buffer("v", torch.zeros(0))

    def reset(self, batch, dim, device, dtype):
        self.v = torch.zeros(batch, dim, device=device, dtype=dtype)

    def forward(self, x):
        B, T, D = x.shape
        if self.v.numel() != B * D:
            self.reset(B, D, x.device, x.dtype)
        spikes = torch.zeros_like(x)
        v = self.v
        for t in range(T):
            v = self.beta * v + x[:, t, :]                       # 积分
            s = SpikeFn.apply(v, self.alpha, self.thresh)        # 发放(替代梯度)
            v = v - s.detach() * self.thresh                     # 软重置(梯度只经发放决策)
            spikes[:, t, :] = s
        self.v = v.detach()                                      # 跨 token 携带到下一序列
        return spikes


class CausalSelfAttn(nn.Module):
    """因果自注意力(softmax)。SNN 里保留为标准注意力，脉冲化放在 FFN / 残差处(SpikFormer 风格)。"""

    def __init__(self, dim, n_heads, block_size):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.register_buffer(
            "mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(self.dim, dim=-1)
        q = q.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        k = k.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class SpikingBlock(nn.Module):
    """Pre-LN 脉冲 Transformer Block：注意力输出→LIF 脉冲(残差)；FFN 中段→LIF 脉冲。"""

    def __init__(self, dim, n_heads, block_size, ff_mult=4, beta=0.9, thresh=1.0, alpha=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = CausalSelfAttn(dim, n_heads, block_size)
        self.lif1 = LifNeuron(beta, thresh, alpha)              # 注意力输出 → 脉冲
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ff_mult * dim, bias=False)
        self.lif2 = LifNeuron(beta, thresh, alpha)              # FFN 中段 → 脉冲
        self.fc2 = nn.Linear(ff_mult * dim, dim, bias=False)

    def forward(self, x):
        a = self.attn(self.norm1(x))
        x = x + self.lif1(a)                                    # 残差 + 脉冲
        h = self.fc2(self.lif2(self.fc1(self.norm2(x))))       # 线性→脉冲→线性
        x = x + h
        return x


class SpikingMiniLM(nn.Module):
    def __init__(self, vocab, dim=64, n_layers=3, n_heads=4, block_size=128,
                 beta=0.9, thresh=1.0, alpha=2.0, ff_mult=4):
        super().__init__()
        self.block_size = block_size
        self.vocab = vocab
        self.tok_emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList(
            [SpikingBlock(dim, n_heads, block_size, ff_mult, beta, thresh, alpha)
             for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0, 0.02)

    def reset_states(self):
        """清空所有 LIF 膜电位，开始一条新序列。"""
        for mod in self.modules():
            if isinstance(mod, LifNeuron):
                mod.v = torch.zeros(0)

    def forward(self, idx):
        # idx: (B,T) token ids
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, tok_ids, max_new=120, temperature=1.0, top_k=0, rep_penalty=1.0):
        """单 token 逐个流入(膜电位跨 token 持续)，自回归生成。"""
        self.reset_states()
        prompt = tok_ids[: self.block_size]
        # 预热：把 prompt 逐 token 流过，建立膜电位
        last_logits = None
        for i in range(prompt.size(0)):
            last_logits = self.forward(prompt[i:i + 1].unsqueeze(0))[:, -1, :]
        out_ids = []
        cur = None
        for _ in range(max_new):
            if cur is None:
                logits = last_logits
            else:
                logits = self.forward(cur)[:, -1, :]
            logits = logits / max(temperature, 1e-6)
            if rep_penalty != 1.0 and out_ids:
                # 对已出现过的 token：正 logits 除以 rp、负 logits 乘 rp（CTRL 论文式）
                seen = torch.tensor(sorted(set(out_ids[-64:])), dtype=torch.long, device=logits.device)
                v = logits[0, seen]
                v = torch.where(v > 0, v / rep_penalty, v * rep_penalty)
                logits[0, seen] = v
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                vk, _ = torch.topk(logits, k)
                logits[logits < vk[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            cur = torch.multinomial(probs, 1)
            out_ids.append(cur.item())
        return out_ids
