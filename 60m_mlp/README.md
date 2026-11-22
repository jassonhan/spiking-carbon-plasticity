# Spiking MiniLM → MLP：60M 中文语言模型训练与碳基可塑性实验

本仓库归档 2026-11 的 60M 中文因果语言模型完整实验线：从脉冲神经网络（SNN）改造、
到消融发现脉冲 FFN 是质量瓶颈、转为 MLP 变体、再到"冻结底座 + 反馈对齐快权重"的
持续学习闭环验证。

## 一句话结论

| 阶段 | val CE（越低越好） |
|---|---|
| SNN（LIF FFN，5 天 5 个 LR 周期） | 6.1506（平台） |
| **MLP（ReLU FFN，7 个周期）** | **4.5711（-1.58，26%）** |
| Stage3 局部学习（冻结底座 + FA 快权重） | 新任务 -0.08 CE，旧任务漂移 +0.007（零遗忘） |

## 目录结构

```
core/           核心模型与训练脚本
  snn_minilm.py         SpikingMiniLM（LIF + 替代梯度 + 门控注意力）
  snn_pretrain_mlp.py   MLP 变体主线训练（自监督 LM，bf16，三线 val）
  stage3_modulator.py   Stage3 基础件（PlasticLinear / ΔW / 资格痕迹）
  mlp_stage3.py         Stage3 接入 MLP（可插拔调质：novelty / token_ce）
  fa_stage3.py          反馈对齐（固定随机反馈矩阵 + 局部 ΔW 更新）
experiments/    实验与评测脚本
  exp_mlp_vs_lif.py            LIF FFN → ReLU 消融（决定性实验）
  eval_probes.py               领域探针（65GB bin 切 16 段测 CE）
  stage3_test_t1*.py           T1 少样本适应（合成 / 真实 / 英语新领域）
  stage3_ab_novelty_vs_tokence.py  调质模式 A/B
  t2_sequence_memory.py        T2 序列记忆 A→B→A
  t2_full_closure.py           4 任务连续学习闭环（总遗忘 + savings）
  t3_data_efficiency.py        T3 数据效率曲线（FA vs 全参微调）
  build_en_tokens.py           英文维基 → BPE token 流
reports/        实验报告
logs/           训练日志（7 周期 MLP 主线）
```

## 关键实验结果

### 1. 消融：LIF FFN → ReLU（exp_mlp_vs_lif.py）
同一权重、同 LR 轨迹、同数据，唯一变量是 FFN 中段 LIF → ReLU：
- 4000 步：val 6.151 → 5.769（-0.381），吞吐 10147 → 29520 tok/s（2.9×）
- 30000 步：5.334；继续 7 个周期到 **4.5711**

### 2. 领域探针（eval_probes.py）
65GB 语料切 16 段逐段测 CE：法律 3.88（已过拟合）、新闻 5.14、通用段均匀 6.1-6.29。
结论：无欠拟合领域，配比加权收益有限。

### 3. Stage3 局部学习（fa_stage3.py）
- 纯 Hebbian 三因子：-0.005（无方向，天花板）
- token 级调质：无改进（证明缺的不是幅度而是方向）
- **反馈对齐**：-0.075（36×），中文漂移仅 +0.017
- **+ dw 衰减 0.02（突触归一化）**：-0.080 / 漂移 +0.007，多遍稳定

| 方法 | 英文改善 | 中文漂移 | 耗时 |
|---|---|---|---|
| FA + decay | -0.080 | **+0.007** | 13s |
| 全参微调 | -0.232 | +0.416 | 34s |

### 4. 持续学习闭环（t2_full_closure.py）
4 任务（英文/新闻/法律/通用）连续学 + 全回访：
- 平均峰值遗忘 **-0.0021**（负值 = 整体略改善）
- 单任务最大退化 +0.007（0.16%）
- savings 1.32×（已学任务重学更快）

## 复现

```bash
# 数据：万卷中文语料 token 流（int32）
#   base 池 + extra 池（.bin / .npy，128 窗口）
# 词表：bpe_chinese_wanjuan.json（8922）

# 训练（60M）
SNN_MLP_STEPS=30000 SNN_MLP_LR=5e-6 SNN_MLP_BATCH=128 \
SNN_MLP_RESUME=snn34_mlp.best.pt python snn_pretrain_mlp.py

# 220M（参数化）
SNN_MLP_TAG=mlp220 SNN_MLP_DIM=1024 SNN_MLP_LAYERS=16 SNN_MLP_HEADS=16 \
SNN_MLP_BATCH=64 SNN_MLP_WARMUP=2000 SNN_MLP_LR=4e-6 python snn_pretrain_mlp.py
```

## 模型权重

`snn34_mlp.best.pt`（245MB，val 4.5711）超过 GitHub 单文件限制，需单独分发
（网盘 / Release 附件 / Git LFS）。

## 硬件

RTX 5070 Ti 16GB。60M @ bs128 = 28000 tok/s；220M @ bs64 = 9852 tok/s（峰值显存 14.1GB）。
