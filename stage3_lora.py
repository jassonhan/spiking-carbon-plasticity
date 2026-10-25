# -*- coding: utf-8 -*-
"""
Stage 3 轮6：LoRA 旁路 + 神经达尔文选择（Darwin-LoRA）。

背景（轮1-5 判负链）：率基 Hebbian 方向正交 -> FA 误差广播方向正交 -> 末层杠杆无效。
结构性结论：冻结 95% + 只动 FFN，任何"固定规则方向"都学不进语言新知识。

轮6 方案的两个转变：
  1. 杠杆集中：每层 fc2 加低秩旁路 ΔW = B·A（r=8，B 零初始化——加载即无扰动，
     bootstrap 底子无损）。328K 参数（0.5%）承载全部可塑性。
  2. 方向来源换血：放弃"共现/误差推导"的规则方向，改用**随机试探 + 奖励选择**
     （Edelman 神经达尔文主义：突触过产出-选择固化）。每步：
       θ' = θ + σ·ξ,  ξ~N(0,1)
       若 CE(θ') < base − margin: 接受（θ = θ'，"这批突触活下来了"）
       否则回滚
     margin/σ 用 1/5 成功律自适应（进化策略经典）。
  局部性叙事：选择信号只有本窗 CE 标量（神经调质），无梯度、无反传链、无跨层信号。

注意：单窗 CE 噪声 ~±0.3，margin 必须大于噪声地板；base 用指数滑动均值。
"""
import math
import torch
import torch.nn as nn


class DarwinLoRA:
    """SpikingMiniLM 的 fc2 低秩旁路 + 神经达尔文选择器（不改模型定义，hook 实现）。"""

    def __init__(self, model, r=8, sigma=0.05, base_alpha=0.05,
                 margin0=0.02, accept_target=0.2, seed=11, verbose=True):
        self.model = model
        self.r = r
        self.sigma = sigma
        self.base_alpha = base_alpha
        self.margin0 = margin0
        self.accept_target = accept_target      # 1/5 成功律目标接受率
        self.verbose = verbose
        self.dev = next(model.parameters()).device
        self._hooks = []
        self._x2 = {}                            # blk_idx -> fc2 输入 (B,T,4D)
        self._A = {}                             # 提案参数（nn.Parameter 语义但手动管理）
        self._B = {}
        self._base_ce = None                     # 指数滑动均值
        self._margin = margin0
        self.stats = []                          # (ce, accepted, margin, sigma)
        self._pending = False
        self.n_accept = 0
        self.n_total = 0
        self._attached = False

    # ---------- 旁路参数与 hook ----------
    def attach(self):
        assert not self._attached
        g = torch.Generator().manual_seed(11)
        for i, blk in enumerate(self.model.blocks):
            d_out, d_in = blk.fc2.weight.shape                   # (D, 4D)
            A = (torch.randn(self.r, d_in, generator=g).to(self.dev) * 0.02)
            B = torch.zeros(d_out, self.r, device=self.dev)
            self._A[i] = A
            self._B[i] = B
            def mk_pre(idx):
                def hook(mod, inp):
                    self._x2[idx] = inp[0].detach()              # (B,T,4D)
                return hook
            def mk_post(idx):
                def hook(mod, inp, out):
                    x2 = self._x2.get(idx)
                    if x2 is None:
                        return out
                    # 旁路：B @ (A @ x2) —— (B,T,4D)->(B,T,r)->(B,T,D)
                    corr = (x2 @ self._A[idx].T) @ self._B[idx].T
                    return out + corr
                return hook
            self._hooks.append(blk.fc2.register_forward_pre_hook(mk_pre(i)))
            self._hooks.append(blk.register_forward_hook(mk_post(i)))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._attached = True
        if self.verbose:
            npar = sum(self._A[i].numel() + self._B[i].numel() for i in self._A)
            print("[darwin-lora] attached: r=%d | bypass params=%d (%.2f%% of model) | sigma=%.3f"
                  % (self.r, npar, 100.0 * npar / 34.3e6, self.sigma))

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attached = False

    # ---------- 达尔文步 ----------
    def propose(self):
        """生成提案 θ' = θ + σ·ξ 并立即生效（下一窗 forward 即评估提案）。"""
        for i in self._A:
            xi_a = torch.randn_like(self._A[i]) * self.sigma
            xi_b = torch.randn_like(self._B[i]) * self.sigma * 0.1   # B 从 0 起步，扰动小些
            self._A[i] = self._A[i] + xi_a
            self._B[i] = self._B[i] + xi_b

    def evaluate(self, ce):
        """提案窗的 CE 到达：accept（固化提案）/ reject（回滚）。返回 (accepted, base)。"""
        if self._base_ce is None:
            self._base_ce = ce
            self.stats.append((ce, False, self._margin, self.sigma))
            return False, self._base_ce
        accepted = ce < self._base_ce - self._margin
        if not accepted:
            # 回滚：需要恢复提案前状态 -> 用镜像技巧：提案时保存旧值太贵？
            # 不贵：A/B 总量 328K，直接保存。
            pass
        self.n_total += 1
        if accepted:
            self.n_accept += 1
        else:
            self._restore()
        # 1/5 成功律自适应
        rate = self.n_accept / max(1, self.n_total)
        if self.n_total % 20 == 0:
            if rate > self.accept_target:
                self._margin = min(self._margin * 1.3, 0.5)      # 太容易接受 -> 提高门槛
                self.sigma = self.sigma * 1.1
            else:
                self._margin = max(self._margin * 0.7, 1e-4)     # 太难 -> 降低门槛
                self.sigma = self.sigma * 0.9
        # base 更新（用实际观测，无论接受与否）
        self._base_ce = (1 - self.base_alpha) * self._base_ce + self.base_alpha * ce
        self.stats.append((ce, accepted, self._margin, self.sigma))
        return accepted, self._base_ce

    def _restore(self):
        """回滚到上一固化状态（提案前的 A/B）。"""
        for i in self._A:
            self._A[i].copy_(self._A_snap[i])
            self._B[i].copy_(self._B_snap[i])

    # 快照（提案前调用）
    def _snapshot(self):
        self._A_snap = {i: self._A[i].clone() for i in self._A}
        self._B_snap = {i: self._B[i].clone() for i in self._B}

    # 便捷：完整一步（propose -> 由外部 forward 得 ce -> evaluate）
    def step_begin(self):
        self._snapshot()
        self.propose()

    def summary(self, last_n=30):
        s = self.stats[-last_n:]
        if not s:
            return "(no stats)"
        import statistics as st
        ce = [x[0] for x in s]
        acc = sum(1 for x in s if x[1]) / len(s)
        return "last %d: CE %.3f->%.3f | accept rate %.0f%% | margin %.3f sigma %.3f" % (
            len(s), ce[0], ce[-1], 100 * acc, self._margin, self.sigma)

    def state_dict(self):
        return {"A": {i: v.cpu() for i, v in self._A.items()},
                "B": {i: v.cpu() for i, v in self._B.items()},
                "base_ce": self._base_ce, "margin": self._margin, "sigma": self.sigma,
                "stats": self.stats}


class HebbLoRA(DarwinLoRA):
    """Hebbian 规则直接作用在 LoRA 低秩参数上（轮7：联想记忆主场任务）。

    与 DarwinLoRA 的区别：不做随机试探/选择，改为轮4 修正版 Hebbian 资格迹直接更新 A/B：
      B += eta * outer(post_dev, h_r)        # (D,r)：旁路隐层活动 h_r=A·x2 与输出偏离共现
      A += eta * outer(h_r - homeo, u)       # (r,4D)：u 与 h_r pairing
    B 非零小初始化（0.01）保证旁路输出可测（轮6 教训：B=0 时扰动不可测）。
    每窗立即结算（δ 信号弱但任务 CE 信号强，联想任务 CE 从 ~8.5 有大下降空间）。
    """

    def __init__(self, model, r=8, eta=0.3, b_init=0.01, homeo=0.10, verbose=True):
        # 自足式初始化（DarwinLoRA 不传递 Plasticizer 属性链，这里手动设齐）
        self.model = model
        self.r = r
        self.eta = eta
        self.b_init = b_init
        self.homeo = homeo
        self.verbose = verbose
        self.dev = next(model.parameters()).device
        self._hooks = []
        self._x2 = {}
        self._post_out = {}
        self._post_run = {}
        self._A = {}
        self._B = {}
        self.stats = []
        self._attached = False

    def attach(self):
        # 复用父类 hook 结构，但 B 非零初始化 + 额外捕获 blk 输出（post 信号）
        assert not self._attached
        g = torch.Generator().manual_seed(11)
        for i, blk in enumerate(self.model.blocks):
            d_out, d_in = blk.fc2.weight.shape
            self._A[i] = (torch.randn(self.r, d_in, generator=g).to(self.dev) * 0.05)
            self._B[i] = (torch.randn(d_out, self.r, generator=g).to(self.dev) * self.b_init)
            def mk_pre(idx):
                def hook(mod, inp):
                    self._x2[idx] = inp[0].detach()
                return hook
            def mk_post(idx):
                def hook(mod, inp, out):
                    x2 = self._x2.get(idx)
                    if x2 is not None:
                        corr = (x2 @ self._A[idx].T) @ self._B[idx].T
                        out = out + corr
                    self._post_out[idx] = out.detach()
                    return out
                return hook
            self._hooks.append(blk.fc2.register_forward_pre_hook(mk_pre(i)))
            self._hooks.append(blk.register_forward_hook(mk_post(i)))
            self._post_out = getattr(self, "_post_out", {})
            self._post_run = getattr(self, "_post_run", {})
            self._post_run[i] = torch.zeros(d_out, device=self.dev)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._attached = True
        if self.verbose:
            npar = sum(self._A[i].numel() + self._B[i].numel() for i in self._A)
            print("[hebb-lora] attached: r=%d | params=%d | eta=%.2f b_init=%.3f"
                  % (self.r, npar, self.eta, self.b_init))

    @torch.no_grad()
    def observe_update(self, token_weight=None):
        """单窗 Hebbian 更新（post=blk 输出偏离稳态, pre=旁路隐层 h_r 与输入 u）。"""
        assert self._attached
        dw_total = 0.0
        for i, blk in enumerate(self.model.blocks):
            x2 = self._x2.get(i)
            po = self._post_out.get(i)
            if x2 is None or po is None:
                continue
            if token_weight is not None:
                w = token_weight.to(x2.device)
                w = w / (w.sum() + 1e-8)
                x2m = torch.einsum("t,btd->bd", w, x2).mean(dim=0)
                post = torch.einsum("t,btd->bd", w, po).mean(dim=0)
            else:
                x2m = x2.mean(dim=(0, 1))
                post = po.mean(dim=(0, 1))
            self._post_run[i] = 0.99 * self._post_run[i] + 0.01 * post
            post_dev = post - self._post_run[i]
            h_r = x2m @ self._A[i].T                      # (r,) 旁路隐层活动
            # B [D,r]: post_dev(D) × h_r(r) —— 输出偏离与旁路活动共现
            # A [r,4D]: h_r(r) × u(4D) —— pairing
            # 限幅：单步更新 ≤ 20% 参数自身范数（LoRA 参数小，eta 必须温和）
            for P, upd in ((self._B[i], torch.outer(post_dev, h_r)),
                           (self._A[i], torch.outer(h_r - self.homeo, x2m))):
                upd = upd / (upd.abs().max() + 1e-8)      # 归一化（轮3 教训：量级失控）
                dn = upd.norm(dim=1, keepdim=True)
                n = P.norm(dim=1, keepdim=True)
                scale = torch.clamp(0.2 * n / (self.eta * dn + 1e-8), max=1.0)
                P.add_(self.eta * upd * scale)
            dw_total += float(self._B[i].norm()) + float(self._A[i].norm())
        self._x2.clear()
        self._post_out.clear()
        self.stats.append(dict(dw=round(dw_total, 5)))
        return dw_total
