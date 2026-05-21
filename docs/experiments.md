# 实验结果与模型档案

> 所有训练实验的设置、结果、解耦评估数据汇总。日常工作记录见 `docs/daily_log.md`，论文方法对应见 `docs/research_context.md`。

---

## 关键模型档案

| 代号 | 路径 | 状态 | 关键指标 |
|---|---|---|---|
| 5/13 baseline | `outputs/20260513_300K_hotlap_baseline/model/final/` | ✅ 历史 baseline | HCSF IM 0.086 ± 0.046, jerk 64.0 ± 1.3 |
| 5/14 F1 OOD | `outputs/20260514_305K_race_OODfail/model/final/` | ⚠️ V_φ 过度激进失败案例 | HCSF jerk 118.7, IM 0.42（仅 OOD 训练参考）|
| 5/19 G2 plateau | `outputs/20260519_G2_1M_plateau/model/final/` | ⚠️ Plateau 4.77 | HCSF IM 0.084 ± 0.062, jerk 61.77 ± 1.78 |
| **🏆 5/21 G5 50K** | `outputs/20260520_G5_1M_pathC_div_50KSOTA/model/checkpoints/step_00050000/` | ✅ **SOTA** | **HCSF IM 0.010 ± 0.007, jerk 60.97 ± 5.3, 干预 1.2 ± 0.9%** |
| 5/21 G5 final | `outputs/20260520_G5_1M_pathC_div_50KSOTA/model/final/` | ❌ 发散，不可用 | v_mean = -417 |

---

## 训练实验明细

### Exp 1: 5/13 Hotlap 300K (单车 baseline)

- **配置**: Hotlap 模式 + 无对手 + 10M_SAC warmup（5s）+ 跳过 INIT + 300K 步
- **训练时长**: ~4h（RTX 3060）
- **V_φ**: 收敛到 ~4
- **解耦评估** (10M as human, noise=0.3, 500 步):
  - HCSF IM 0.075, jerk 60.3, 干预 8.6%, no-cand 10 次
  - 符合论文 H2（HCSF jerk < LRSF）

### Exp 2: 5/14 F1 Race 305K (含对手, OOD 失败)

- **配置**: Quick Race 模式 + 1 Mazda 对手 + 10M_SAC warmup + 跳过 INIT + 启用对手 g
- **训练时长**: ~4h
- **失败模式**: Race grid 起步对 10M Hotlap warmup driver OOD → buffer 99% stationary → V_φ 学到"几乎所有状态危险"
- **解耦评估** (10M as human, noise=0.3, 300 步):

  | Filter | jerk | IM | 干预率 | speed mean |
  |---|---|---|---|---|
  | None | 70.6 | 0 | 0% | 4.1 m/s |
  | LRSF | 56.5 | 0.02 | 1.6% | 9.1 m/s |
  | **HCSF** | **118.7** | **0.42** | **26.6%** | 9.6 m/s |

- **结论**: HCSF 比 LRSF 还差 2x（违反 H2），大量 "no candidate satisfies Q-CBF" → 粗暴 fallback
- **教训**: OOD 是根因，单纯调参没用；存档为反面教材

### Exp 3: 5/19 G2 Hotlap 1M (Plateau)

- **配置**: Hotlap + 无对手 + 10M warmup(5s) + 跳过 INIT + 1M 步
- **训练时长**: ~12h
- **失败模式**: V_φ 从 2.1 → 4.0（前 150K）→ 4.77 plateau（剩 850K 步停滞）
- **根因**: Hotlap 单车 g(x) 始终 > 0 → V_φ Bellman target `min{g, Q}` 总被 Q 主导 → V_φ 退化为 Q 影子
- **解耦评估** (10M as human, noise=0.3, 500 步):

  | 指标 | 300K (5/13) | 1M (5/19) |
  |---|---|---|
  | HCSF IM | **0.075** | 0.156 |
  | HCSF jerk | **60.3** | 65.5 |
  | HCSF 干预率 | 8.6% | 18.6% |
  | no-candidate | 10 次 | 4 次 |

- **结论**: 1M 不比 300K 好。瓶颈不在步数，在训练场景多样性

### Exp 4: 5/20 G3 Hotlap 1M + INIT (612K kill)

- **配置**: Hotlap + INIT 恢复（Q_init_term=2.0, T_init_max=3s）+ π^♦ 驱动 + 1M 步
- **kill 原因**: speed 单调下降 20→13 m/s，63% episode "Speed too low" 终止，跟 G2 同 plateau 模式
- **教训**: 仅恢复 INIT 不够，π^♦ 在 Hotlap 下仍 safe-slow

### Exp 5: 5/20 G4 Option 2 (10M 驱动 TRAINING, 122K kill, dead end)

- **配置**: 改 `agent.py` 让 10M_SAC 驱动 TRAINING + 噪声 0.1（误读 paper §V-D intent）
- **kill 原因**: 10M 太能开 → episode 跑满 15001 步 timeout → INIT ratio 0.02%（5/19 是 1-4%）→ V_φ stuck @ 2.4
- **教训**: 10M 不会失败 → 永远不触发 episode reset → INIT 几乎不点火。**架构错误，回退**

### Exp 6: 5/21 G5 Path C (短 episode hack, 1M 完成, 50K SOTA)

- **配置**:
  - π^♦ 驱动 TRAINING（paper-aligned 回退选项 1）
  - `max_episode_py_time: 600s → 60s`（强制 10x reset → 10x INIT 触发）
  - INIT 恢复（同 G3）
  - 1M 步
- **训练时长**: ~12h（过夜）
- **训练曲线 V_φ 发散**:

  | step | v_loss | v_mean | 状态 |
  |---|---|---|---|
  | 45K | 0.054 | **2.44** | ✅ 健康 |
  | 91K | 0.084 | 1.64 | 下降 |
  | 136K | 3.29 | -2.27 | 转负 |
  | 181K | 1,802 | -83 | 爆炸 |
  | 362K | 337M | **-18,228** | 峰值崩溃 |
  | 1M | 24K | -417 | 稳定坏值 |

- **发散机制**: actor-critic 反馈循环爆炸——INIT 引入极端负 Q → V_φ Bellman target 跟着负 → π^♦ max-Q 用炸了的 Q 训练 → 反向放大
- **后 95% 步数在毒化网络上训练，浪费了**
- **关键产出**: 50K checkpoint (V_φ 还未发散) 是迄今最优 HCSF 模型
- **下一步修复**: V_φ target clipping + 降 v_lr + early stopping

---

## SOTA 实证：G5 50K Variance Eval（9 次复测，2026-05-21）

### 评估配置

- Human policy: 10M_SAC (`AssettoCorsaGymDataSet/.../20240404_SAC_10M/model/final`)
- noise=0.3, steps=500, 解耦模式
- 每个模型测 3 次（独立采样 noise + HCSF 候选）

### 9 次完整数据（HCSF filter 部分）

| Model | Run | IM | jerk | 干预率 | V_avg | LRSF jerk |
|---|---|---|---|---|---|---|
| **G5 50K** | r1 | 0.0179 | 60.0 | 1.6% | 2.4 | 66.8 |
| | r2 | 0.0120 | 67.9 | 1.8% | 2.3 | 61.1 |
| | r3 | 0.0001 | 55.0 | 0.2% | 2.4 | 65.3 |
| **5/13 baseline** | r1 | 0.1502 | 63.3 | 14.4% | 2.0 | 66.2 |
| | r2 | 0.0502 | 62.9 | 5.8% | 2.4 | 63.5 |
| | r3 | 0.0560 | 65.8 | 4.8% | 2.3 | 64.1 |
| **5/19 G2 plateau** | r1 | 0.0254 | 61.1 | 4.8% | 2.7 | 65.0 |
| | r2 | 0.0567 | 60.0 | 6.2% | 2.5 | 62.6 |
| | r3 | 0.1702 | 64.2 | 20.0% | 2.5 | 61.3 |

### 统计聚合（mean ± std, n=3）

| 指标 | G5 50K | 5/13 baseline | 5/19 G2 plateau | G5 是否显著更好 |
|---|---|---|---|---|
| HCSF **IM** | **0.0100 ± 0.0074** | 0.0855 ± 0.046 | 0.0841 ± 0.062 | ✅ **8x 优势，区间不重叠** |
| HCSF **jerk** | **60.97 ± 5.31** | 64.0 ± 1.28 | 61.77 ± 1.78 | ⚠️ 均值最低，方差较大 |
| HCSF **干预率** | **1.2 ± 0.9%** | 8.3 ± 5.3% | 10.3 ± 8.4% | ✅ **7-9x 更低** |
| 平均 LRSF jerk | 64.4 | 64.6 | 63.0 | — |

### 关键判定

**G5 50K = 真 SOTA，不是偶然**：

1. **IM 区间完全不重叠**: G5 max 0.0179 < 5/13 min 0.0502 → 统计显著
2. **干预率区间完全不重叠**: G5 max 1.8% < baseline min 4.8% → V_φ 边界更精确
3. **3 次复测都低于 5/13 baseline 均值**: r1/r2/r3 IM 全部 < 0.02 < 0.086
4. **原版 0.0074 是典型值**: 落在复测 [0.0001, 0.0179] 区间内，不是异常好的 outlier

**警告**：
- **jerk 方差较大**（G5 r2: 67.9 反例 vs r3: 55.0），论文级图表必须报 mean±std
- 论文 H2（HCSF jerk < LRSF jerk）3 次中 2 次满足，均值满足（60.97 < 64.4）

### 数据文件位置

- 9 个 metrics.csv: `/tmp/ac_eval_hcsf/variance_*_metrics.csv`
- 完整 log: `/tmp/variance_eval.log`

---

## 路径性能纵向对比（HCSF SOTA 演化）

| 实验 | 步数 | HCSF IM | HCSF jerk | 干预率 | 备注 |
|---|---|---|---|---|---|
| 5/14 F1 | 305K | 0.42 | 118.7 | 26.6% | OOD 失败 |
| 5/19 G2 | 1M | 0.156 | 65.5 | 18.6% | Plateau |
| 5/13 baseline | 300K | 0.075 | 60.3 | 8.6% | 之前 SOTA |
| **🏆 5/21 G5 50K** | **50K** | **0.0074** | **59.9** | **1.2%** | **新 SOTA** |

**G5 50K 用 1/6 步数（50K vs 300K）实现 10x 更低 IM。** 关键在 Path C 的短 episode hack 让 V_φ 在 50K 步内就拿到足够边界数据。

---

## 下一步实验候选

1. **修 V_φ 发散后重训**: target clipping + 降 v_lr + early stopping，再跑 1M-3M 步
2. **配置对齐论文**: batch_size 128→256, memory_size 8M→20M
3. **完整 12.8M 步训练**: 在 JHU 实验室 GPU 上跑（3060 也可，1-2 周）
4. **赛道切换**: 银石（如有时间）
5. **真人用户研究**: 复现论文 Fig 5/7-9（JHU 期间）
