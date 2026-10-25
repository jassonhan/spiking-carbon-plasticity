# SpikingMiniLM 碳基可塑性运营栈 / Spiking Carbon Plasticity

**把一个 34M 脉冲语言模型的学习方式从「反向传播训练」切换到「局部可塑性规则运营」——完整实验栈与全部阴性/阳性结果。**

> Switching a 34M spiking language model from BPTT training to *locally-plastic online operation*: the full experiment stack, with every positive and negative result reported.

[English abstract below](#english-abstract)

---

## 这是什么

本项目将 MiniLM 因果语言模型改造为脉冲神经网络（**SpikingMiniLM**：attention 骨干保留，FFN 替换为 LIF 脉冲神经元），并回答一个核心问题：

> **冻结反向传播之后，这个网络还能不能继续学习？**

这是「类碳基学习」路线的 Stage 3：先用监督学习把语言底子教会（bootstrap），然后**冻结发育，日常学习全部交给局部可塑性规则**——正如生物先有发育编程、再有经验学习。胜负手不是静态 ppl，而是**持续在线学习的保持与数据效率**。

## 核心发现

### ① 学习规则：八轮排除法收敛到 LMS delta rule

| 规则 | 方向信息 | 写入新知识 | 判决 |
|---|---|---|---|
| 率基 Hebbian / R-STDP | 共现（无方向） | ✗ | 方向与 CE 梯度余弦 ≈ 0.028 |
| Feedback Alignment 误差广播 | 随机投影 | ✗ | 余弦 ≈ 0.005 |
| Darwin 随机试探 + 奖励选择 | 标量奖励 | ✗ | 零阶信噪比不足 |
| **LMS（输出层 delta rule）** | **预测误差 ⊗ 局部活动** | **✓ CE −41%** | **唯一成功** |
| BPTT（对照） | 全局梯度 | ✓ CE −40% | 参照系 |

**方向性诊断**（`stage3_direction_diag.py`）是本项目的方法论支柱：对同一窗口计算局部规则更新与 BPTT 梯度的余弦相似度，用 |cos| ≈ 0.01–0.03 的硬数据关闭了所有"调参就能通"的幻想，把问题逼到规则选型层面。

**LMS = softmax − onehot ⊗ h**：误差信号与输入活动都在输出层本地，无任何反传链——即小脑的 Marr-Albus 规则。它达到了 BPTT 写入效果的 ~80%。

### ② 稳定机制：重放抗遗忘 + 60% savings

序列记忆测试 A→B→A（`stage3_t2_sequence.py`）：

| 条件 | 旧任务遗忘 | 新任务学习 |
|---|---|---|
| 无重放 | +4.02 CE（比基线还差） | ✓ |
| **海马重放 20%** | **+1.53（减 62%）** | ✓（近零代价） |

重学 savings ≈ **60%**（三跑稳定）——突触残留痕迹真实存在，Ebbinghaus 式节省效应在 LMS 下复现。

### ③ 破局：任务门控稀疏输出（海马索引架构）⭐

四任务顺序流的漂移地板（+1.5 CE/任务）证明：共享输出层的干扰是**表示冲突**而非噪声——门控睡眠再巩固实验中，不存在"帮一个任务而不伤另一个"的局部更新方向。

破局架构（`stage3_gated_gen.py`）：

```
底座 lm_head 冻结            →  皮层长期存储（不被覆写）
每任务私有增量 ΔW_t          →  经验私有痕迹（闭环 LMS 学习）
情境键 WTA 硬门控            →  海马索引 + 模式补全
多索引（每窗全窗h+前缀h）     →  多痕迹情境索引
```

**四任务流实证（800 步/任务）**：B/C/D 漂移 **0.000**，平均漂移 **+0.18 vs 无门控地板 +1.9（10 倍改善）**。闭环私有 delta 的学习质量反超共享 LMS（干净梯度信号无干扰）。

### ④ 已量化的边界（诚实报告）

- LMS 碎片级渗透在 ~1000 步饱和，生成 recall 为词片段级——34M + 输出层杠杆的容量上限
- 生成期路由依赖情境键区分度：稠密 h 空间的任务键分布重叠（缺 pattern separation）——这是下一场的理论切入点
- 替代梯度架构级不稳定：60M 扩容在 LR 1e-5 下 gnorm 仍爆 1e4，容量路线冻结

## 快速开始

```bash
pip install torch numpy

# 1. 冒烟：门控四任务流（60 步/任务，CPU 几分钟）
python stage3_gated_stream.py --smoke

# 2. 正式：门控流（800 步/任务 + 保持分析，GPU ~12 分钟）
python stage3_gated_stream.py --steps-per-task 800

# 3. 序列记忆 A→B→A（遗忘/savings 分析）
python stage3_t2_sequence.py --stepsA 800 --stepsB 800 --stepsReA 800

# 4. 方向性诊断（判决局部规则与 BPTT 梯度的余弦）
python stage3_direction_diag.py --nwin 8 --dev cpu

# 5. LMS 写入 + best 快照
python stage3_t1_lms.py --steps 2000 --lr 0.5
```

> 预训练权重（34M，val 5.244）与本仓库分词器配套，因体积未随仓库分发——见 [模型权重](#模型权重)。

## 仓库结构

```
├── snn_minilm.py              SpikingMiniLM 模型（LIF-FFN 混合架构 + generate/rp 解码）
├── bpe_tokenizer.py           BPE 分词器（+ bpe_chinese_wanjuan.json 词表）
├── stage3_plastic.py          可塑性运营层：Hebbian/R-STDP 资格迹 + LMS + surprise 加权
├── stage3_gated_gen.py        ⭐ GatedLMS：私有增量 + 多索引情境门控（旗舰）
├── stage3_gated_stream.py     门控四任务流（CE 保持验证）
├── stage3_lora.py             Darwin-LoRA / Hebb-LoRA 变体
├── stage3_t1_*.py             T1 少样本适应套件（6 种规则对照）
├── stage3_t2_sequence.py      T2 序列记忆 A→B→A（遗忘/savings）
├── stage3_t4_interleave.py    T4 并行干扰稳定性
├── stage3_stream4.py          四任务长程流（含门控睡眠再巩固）
├── stage3_direction_diag.py   方向性诊断（规则 × BPTT 梯度余弦）
├── stage3_fa_diag.py          FA 方向质量诊断
├── decode_ab_test.py          解码策略 A/B（repetition penalty）
├── rate_samples.py            人眼评分工具（流畅/切题 1-5 锚点）
└── results/                   全部关键实验的原始报告
```

## 结果报告

`results/` 内为各实验的原始输出，核心数字：

- `stage3_t1_lms_report.txt` — LMS 首次写入：CE 8.35→4.90，recall 0→5
- `stage3_t2_sequence_report.txt` — 重放抗遗忘 62% + savings 59%
- `stage3_t4_interleave_report.txt` — 并行学习零遗忘
- `stage3_gated_stream_report.txt` — 门控四任务流平均漂移 +0.18
- `stage3_gated_gen_report.txt` — 门控下 CE 保持 +0.000 复现 + 生成路由边界

## 模型权重

34M SpikingMiniLM（教师蒸馏 27k 步 + v4-full 19408 窗末轮，val 5.244 / ppl 231）未随仓库分发（133MB）。如需复现 T1/T2 的"从预训练权重出发"设定，请提 Issue。

也可以从零训练：`snn_distill_5070.py` 风格的自监督/蒸馏脚本不在本仓库范围（属于 Stage 2 资产），但 `snn_minilm.py` 的模型定义可直接用。

## 路线图

- [ ] 判别性情境键（pattern separation：LDA 式/对比式键变换）
- [ ] 门控下的完整生成评估 + 人眼评分整合
- [ ] T5 元可塑性（先固底是否加快后续学习）
- [ ] 教师蒸馏管线（Stage 2 资产整理）
- [ ] 替代梯度改进（atan/triangle 系）解锁扩容

## English abstract

We retrofit a 34M causal language model (SpikingMiniLM: attention backbone + LIF spiking FFN) for **post-BPTT online learning** with local plasticity rules. Through eight controlled rule-candidate experiments — including a *direction diagnosis* comparing each candidate's update against the BPTT gradient (cosine ≈ 0.01–0.03 for all Hebbian/FA/evolutionary variants) — we converge on the **output-layer LMS delta rule** (softmax − onehot) ⊗ h, the only local rule that writes new knowledge (−41% CE, ~80% of BPTT's effect, zero backprop chains; the cerebellar Marr-Albus rule). A hippocampus-inspired **gated sparse output** (frozen base + per-task private deltas + multi-index context keys with winner-take-all routing) reduces catastrophic forgetting in a 4-task stream from +1.9 to **+0.18 CE per task** (B/C/D: exactly 0.000), while replay-based consolidation cuts forgetting by 62% in the shared-weight regime with ~60% relearning savings. All negative results, safety-mechanism lessons (zero-init deltas frozen by norm clamps), and capacity boundaries are reported.

## 引用 / Citation

如果你用了本项目的方向性诊断方法或门控可塑性架构，欢迎引用：

```bibtex
@misc{spiking-carbon-plasticity,
  title  = {SpikingMiniLM Carbon Plasticity: Local Learning Rules and Gated Sparse Output for Post-BPTT Online Learning},
  author = {jassonhan and WorkBuddy Agent},
  year   = {2026},
  url    = {https://github.com/jassonhan/spiking-carbon-plasticity}
}
```

## 许可

MIT License
