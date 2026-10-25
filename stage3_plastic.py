# -*- coding: utf-8 -*-
"""
Stage 3 核心框架：SpikingMiniLM 的局部可塑性运营层（C 方案：先教后放）。

设计原则（对应 Stage3_碳基可塑性路线.md）：
  - 模型定义零改动：final best 权重原样加载，推理路径不变
  - 可塑性是外挂：hook 捕获脉冲事件 -> eligibility trace 积累 -> 奖励结算（两相协议）
  - 梯度隔离：全部 no_grad + .data 原地更新，autograd 图不存在
  - 局部性：Δw 只由突触两侧的活动（pre/post trace）+ 全局标量奖励决定，无反传

可塑性突触选择：
  - fc2 (4D -> D)：pre = lif2 输出脉冲 s2（0/1 事件，天然的突触前脉冲），post = block 输出（突触后活动）
  - fc1 (D -> 4D)：pre = norm2 输出（连续活动），post = lif2 脉冲 s2（脉冲 pairing 规则）
  - attention/lm_head 不动：语言底子保留，可塑性集中在 FFN（知识存储层）

规则（率基 STDP + 奖励调制两相协议，Izhikevich 2007 风格）：
  相1（每窗流过后积累资格迹）：
    e_fc2 = (pre_rate + eps) ⊗ (post_dev)                # Hebbian 项
          - homeo * pre_rate ⊗ post_run_mean              # 稳态项（防正反馈爆炸）
    e_fc1 = (pre_u_rate) ⊗ (s2_rate - homeo)             # pairing 项
    E <- beta_e * E + e                                   # 资格迹跨窗衰减（经验记忆）
  相2（奖励信号 = 窗 CE 相对滑动基线的改善）：
    r = tanh((base - ce) / scale)
    w += eta * r * E（Oja 式行归一稳定）

  pre_trace 跨窗持续衰减（beta_tr），模拟"记忆窗"内的时间局部性。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Plasticizer:
    """SpikingMiniLM 的外挂局部可塑性运营器。

    用法：
        pl = Plasticizer(model, eta=2e-3)
        pl.attach()                      # 开启运营（hook + trace）
        for window in windows:           # 逐窗流入（model 常规 forward）
            ce = ce_of(window)
            pl.observe()                 # 相1：积累资格迹
            pl.set_reward(ce)            # 相2：奖励结算（内部自动算基线）
        pl.detach()                      # 冻结（纯推理）
    """

    def __init__(self, model, eta=2e-3, beta_e=0.90, beta_tr=0.90,
                 homeo=0.10, reward_scale=1.0, eps=1e-3, verbose=True):
        self.model = model
        self.eta = eta
        self.beta_e = beta_e
        self.beta_tr = beta_tr
        self.homeo = homeo
        self.reward_scale = reward_scale
        self.eps = eps
        self.verbose = verbose
        self.dev = next(model.parameters()).device
        self._hooks = []
        self._events = {}     # blk_idx -> dict(s2_rate, u_rate, post)
        self._E = {}          # blk_idx -> dict(E_fc1, E_fc2)
        self._w0 = {}         # blk_idx -> dict(fc1, fc2) 初始权重快照（安全网回滚参照）
        self._w0n = {}        # blk_idx -> dict(fc1, fc2) 初始行范数
        self._post_run = {}   # blk_idx -> post 活动的运行均值（稳态参考）
        self._ce_hist = []    # 奖励基线滑动窗口
        self._attached = False
        self.stats = []       # 每窗记录（步数, ce, reward, dw_norm）

    # ---------------- hooks ----------------
    def attach(self):
        assert not self._attached
        for i, blk in enumerate(self.model.blocks):
            def mk_pre_fc1(idx):
                def hook(mod, inp):
                    # fc1 输入 = norm2 输出 u, (B,T,D)
                    u = inp[0].detach()
                    self._events.setdefault(idx, {})["u_rate"] = u.mean(dim=(0, 1))
                    self._per_tok.setdefault(idx, {})["u"] = u
                return hook
            def mk_post_lif2(idx):
                def hook(mod, inp, out):
                    # lif2 输出脉冲 s2, (B,T,4D) —— 0/1 事件
                    s2 = out.detach()
                    self._events.setdefault(idx, {})["s2_rate"] = s2.mean(dim=(0, 1))
                    self._per_tok.setdefault(idx, {})["s2"] = s2
                return hook
            def mk_post_blk(idx):
                def hook(mod, inp, out):
                    # block 输出（fc2 突触后活动）
                    o = (out if not isinstance(out, tuple) else out[0]).detach()
                    self._events.setdefault(idx, {})["post"] = o.mean(dim=(0, 1))
                    self._per_tok.setdefault(idx, {})["post"] = o
                return hook
            self._hooks.append(blk.fc1.register_forward_pre_hook(mk_pre_fc1(i)))
            self._hooks.append(blk.lif2.register_forward_hook(mk_post_lif2(i)))
            self._hooks.append(blk.register_forward_hook(mk_post_blk(i)))
            self._E[i] = {"fc1": torch.zeros_like(blk.fc1.weight.data),
                          "fc2": torch.zeros_like(blk.fc2.weight.data)}
            # per-token 事件缓存（surprise 加权资格迹用；mean 版保留兼容）
            self._per_tok = {}    # blk_idx -> dict(s2[B,T,4D], u[B,T,D], post[B,T,D])
            # 安全网参照：初始权重快照 + 初始行范数
            self._w0[i] = {"fc1": blk.fc1.weight.data.detach().cpu().clone(),
                           "fc2": blk.fc2.weight.data.detach().cpu().clone()}
            self._w0n[i] = {"fc1": blk.fc1.weight.data.norm(dim=1).detach().cpu().clone(),
                            "fc2": blk.fc2.weight.data.norm(dim=1).detach().cpu().clone()}
            d_out = blk.fc2.weight.shape[0]
            self._post_run[i] = torch.zeros(d_out, device=self.dev)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._attached = True
        if self.verbose:
            print("[plastic] attached: %d blocks, eta=%.1e beta_e=%.2f homeo=%.2f"
                  % (len(self.model.blocks), self.eta, self.beta_e, self.homeo))

    def detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attached = False
        if self.verbose:
            print("[plastic] detached (frozen)")

    # ---------------- 相1：资格迹积累 ----------------
    @torch.no_grad()
    def observe(self, token_weight=None):
        """token_weight: (T,) 每位置的 surprise 权重（如归一化的 per-token CE）。
        提供时资格迹按权重积累——预测误差驱动写入（RPE 式可塑性）；
        缺省时退化为窗平均（旧行为）。"""
        assert self._attached and self._events, "call after a forward pass"
        for i, blk in enumerate(self.model.blocks):
            ev = self._events.get(i)
            if not ev or "s2_rate" not in ev:
                continue
            s2 = ev["s2_rate"]            # (4D,) pre of fc2 / post of fc1
            u = ev.get("u_rate")          # (D,)  pre of fc1
            post = ev.get("post")         # (D,)  post of fc2
            if post is None:
                continue
            if token_weight is not None and i in self._per_tok:
                pt = self._per_tok[i]
                if "s2" in pt and "u" in pt and "post" in pt:
                    # per-token 外积的加权平均（RPE 驱动）
                    w = token_weight.to(pt["s2"].device)            # (T,)
                    w = w / (w.sum() + 1e-8)
                    s2w = torch.einsum("t,btd->bd", w, pt["s2"]).mean(dim=0)    # (4D,)
                    uw = torch.einsum("t,btd->bd", w, pt["u"]).mean(dim=0)      # (D,)
                    postw = torch.einsum("t,btd->bd", w, pt["post"]).mean(dim=0)  # (D,)
                    s2, u, post = s2w, uw, postw
            # 稳态参考（运行均值，慢速跟随）
            self._post_run[i] = 0.99 * self._post_run[i] + 0.01 * post
            post_dev = post - self._post_run[i]              # 偏离稳态的部分才可塑
            # fc2.weight [D,4D]=[out,in]: pre=s2(4D), post=post_dev(D) —— Hebbian；稳态项由 post_dev 天然承担
            e2 = torch.outer(post_dev, s2 + self.eps)          # (D,4D)
            # fc1.weight [4D,D]: pre=u(D), post=s2(4D) —— pairing（连续 pre × 脉冲 post）
            e1 = torch.outer(s2 - self.homeo, u)               # (4D,D)
            # 资格迹归一化（关键：outer 量级随活动幅值漂移，必须归一，否则 dw 爆炸/安全网饱和）
            e2 = e2 / (e2.abs().max() + 1e-8)
            e1 = e1 / (e1.abs().max() + 1e-8)
            # 资格迹衰减 + 累加
            self._E[i]["fc2"].mul_(self.beta_e).add_(e2, alpha=1.0)
            self._E[i]["fc1"].mul_(self.beta_e).add_(e1, alpha=1.0)
        self._events.clear()
        self._per_tok.clear()

    # ---------------- 相2：奖励结算 ----------------
    @torch.no_grad()
    def set_reward(self, ce, novelty=0.0):
        """ce: 当前窗 next-token CE。奖励 = 相对滑动基线的改善 + 新奇性加成（探索期）。
        安全网：行范数钳制（≤1.5×初始）+ nan 检测回滚——局部规则无全局优化器兜底，必须自保。"""
        if len(self._ce_hist) >= 5:
            base = sum(self._ce_hist[-50:]) / len(self._ce_hist[-50:])
            r = max(-1.0, min(1.0, (base - ce) / self.reward_scale))
        else:
            base, r = ce, 0.0                      # 冷启动：第一窗无相对奖励
        r = max(-1.0, min(1.0, r + novelty))       # novelty bonus：新经验值得探索（碳基好奇心）
        self._ce_hist.append(ce)
        if len(self._ce_hist) > 200:
            self._ce_hist = self._ce_hist[-100:]
        dw_total = 0.0
        clamped = 0
        if abs(r) > 1e-6:
            for i, blk in enumerate(self.model.blocks):
                for name, dw in (("fc1", self.eta * r * self._E[i]["fc1"]),
                                 ("fc2", self.eta * r * self._E[i]["fc2"])):
                    w = getattr(blk, name).weight.data
                    # 事前限幅：单窗 dw 的行范数增量 ≤ 5% 当前行范数（温和写入，方向不被钳制扭曲）
                    n = w.norm(dim=1, keepdim=True)
                    dn = dw.norm(dim=1, keepdim=True)
                    scale = torch.clamp(0.05 * n / (dn + 1e-8), max=1.0)
                    dw = dw * scale
                    clamped += int((scale < 1.0).sum())
                    w.add_(dw)
                    dw_total += float(dw.norm())
                    # nan 兜底（理论上限幅后不会再触发）
                    bad_rows = ~torch.isfinite(w).all(dim=1)
                    if bad_rows.any():
                        w[bad_rows] = self._w0[i][name][bad_rows].to(w.device)
                        clamped += int(bad_rows.sum())
        if clamped and self.verbose:
            print("[plastic] safety net: %d row-events clamped/rolled-back" % clamped)
        self.stats.append(dict(ce=round(ce, 4), reward=round(r, 4), dw=round(dw_total, 5)))
        return r, dw_total

    # ---------------- 工具 ----------------
    def summary(self, last_n=20):
        s = self.stats[-last_n:]
        if not s:
            return "(no stats)"
        import statistics as st
        ce = [x["ce"] for x in s]; rw = [x["reward"] for x in s]
        return "last %d windows: CE %.3f->%.3f | mean reward %+.4f" % (
            len(s), ce[0], ce[-1], st.mean(rw))

    def state_dict(self):
        return {"E": {i: {k: v.cpu() for k, v in d.items()} for i, d in self._E.items()},
                "post_run": {i: v.cpu() for i, v in self._post_run.items()},
                "ce_hist": self._ce_hist, "stats": self.stats}

    def load_state_dict(self, sd):
        self._E = {i: {k: v.to(self.dev) for k, v in d.items()} for i, d in sd["E"].items()}
        self._post_run = {i: v.to(self.dev) for i, v in sd["post_run"].items()}
        self._ce_hist = sd.get("ce_hist", [])
        self.stats = sd.get("stats", [])


# ---------------- 通用工具 ----------------
@torch.no_grad()
def window_ce(model, x_ids, y_ids, return_per_tok=False):
    """单个窗口的 next-token CE（教师窗约定：label[t]==input[t]，只算 continuation 段）。
    return_per_tok=True 时返回 (ce_scalar, per_tok_ce[T])——per-token 预测误差（RPE 权重）。"""
    import torch.nn.functional as F
    model.eval()
    logits = model(x_ids.unsqueeze(0))            # (1,T,V)
    T = x_ids.shape[0]
    if T <= 1:
        return (float("nan"), None) if return_per_tok else float("nan")
    lg = logits[0, :-1, :]
    tg = y_ids[1:]
    m = tg != -100
    if m.sum() == 0:
        return (float("nan"), None) if return_per_tok else float("nan")
    ce_vec = F.cross_entropy(lg, tg, reduction="none")     # (T-1,)
    ce = float(ce_vec[m].mean())
    if not return_per_tok:
        return ce
    # 对齐回 T 维：位置 t 的 CE = 预测 token t 的误差（logits[t-1] -> target[t]）
    per = torch.zeros(T, device=x_ids.device)
    idx = torch.arange(1, T, device=x_ids.device)[m]       # continuation 位置
    per[idx] = ce_vec[m]
    return ce, per


class FAPlasticizer(Plasticizer):
    """Feedback-Alignment 误差广播版可塑性（Stage 3 轮5+ 的正式规则）。

    与率基 Hebbian 版（Plasticizer）的本质区别：写入方向携带预测误差信息——
      δ_out (T,V) = softmax(logits) − onehot(target)        # 输出层局部误差
      δ_pool (D,) = Σ_t w_t · (δ_out @ E_emb)               # 经冻结 embedding 投影（语义方向）
      Δw_fc2    = η · outer(δ_pool, s2_rate)                # (D,4D)
      δ_fc1     = FA1 @ δ_pool                              # 固定随机广播（无反传链）
      Δw_fc1    = η · outer(δ_fc1, u_rate)                  # (4D,D)
    无资格迹两相（δ 本身含方向与幅度），安全网限幅沿用；梯度隔离同款。
    """

    def __init__(self, model, eta=0.2, fa_scale=0.05, fa_seed=7, **kw):
        super().__init__(model, eta=eta, **kw)
        self.fa_scale = fa_scale
        g = torch.Generator().manual_seed(fa_seed)
        self._FA1 = {}
        for i, blk in enumerate(self.model.blocks):
            d4 = blk.fc1.weight.shape[0]
            d = blk.fc1.weight.shape[1]
            self._FA1[i] = torch.randn(d4, d, generator=g).to(self.dev) * fa_scale

    @torch.no_grad()
    def observe_error(self, logits, target, token_weight=None):
        """单窗结算：logits (T,V)，target (T,)（含 -100 忽略位）。"""
        assert self._attached
        T, V = logits.shape
        probs = F.softmax(logits.float(), dim=-1)                     # (T,V)
        onehot = torch.zeros_like(probs)
        m = target != -100
        onehot[m, target[m]] = 1.0
        delta = probs - onehot                                        # (T,V)
        if token_weight is not None:
            w = token_weight.to(logits.device)
            w = w / (w.sum() + 1e-8)
            dpool = (delta * w.unsqueeze(1)).sum(dim=0)               # (V,)
        else:
            dpool = delta[m].mean(dim=0)                              # continuation 均值
        dpool = dpool @ self.model.tok_emb.weight.float()             # (D,) 语义方向投影
        dw_total = 0.0
        clamped = 0
        for i, blk in enumerate(self.model.blocks):
            ev = self._events.get(i, {})
            if "s2_rate" not in ev:
                continue
            s2 = ev["s2_rate"]
            u = ev.get("u_rate")
            dw2 = self.eta * torch.outer(dpool, s2)                   # (D,4D)
            dfc1 = self._FA1[i] @ dpool                               # (4D,)
            dw1 = self.eta * torch.outer(dfc1, u)                     # (4D,D)
            for name, dw in (("fc2", dw2), ("fc1", dw1)):
                w_ = getattr(blk, name).weight.data
                n = w_.norm(dim=1, keepdim=True)
                dn = dw.norm(dim=1, keepdim=True)
                scale = torch.clamp(0.05 * n / (dn + 1e-8), max=1.0)
                dw = dw * scale
                clamped += int((scale < 1.0).sum())
                w_.add_(dw)
                dw_total += float(dw.norm())
        self._events.clear()
        self._per_tok.clear()
        self.stats.append(dict(dw=round(dw_total, 5), clamped=clamped))
        return dw_total
