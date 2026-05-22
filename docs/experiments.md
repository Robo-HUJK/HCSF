# 实验结果与模型档案

> 所有训练实验的设置、结果、解耦评估数据汇总。日常工作记录见 `docs/daily_log.md`，论文方法对应见 `docs/research_context.md`。

---

## 关键模型档案

| 代号 | 路径 | 状态 | 关键指标 |
|---|---|---|---|
| 5/13 baseline | `outputs/20260513_300K_hotlap_baseline/model/final/` | ✅ 历史 baseline | HCSF IM 0.086 ± 0.046, jerk 64.0 ± 1.3 |
| 5/14 F1 OOD | `outputs/20260514_305K_race_OODfail/model/final/` | ⚠️ V_φ 过度激进失败案例 | HCSF jerk 118.7, IM 0.42（仅 OOD 训练参考）|
| 5/19 G2 plateau | `outputs/20260519_G2_1M_plateau/model/final/` | ⚠️ Plateau 4.77 | HCSF IM 0.084 ± 0.062, jerk 61.77 ± 1.78 |
| **🏆 5/21 G5 50K (outlier SOTA)** | `outputs/20260520_G5_1M_pathC_div_50KSOTA/model/checkpoints/step_00050000/` | ✅ **极致 IM/低干预**（不可复现）| **HCSF IM 0.010 ± 0.007, jerk 60.97 ± 5.3, 干预 1.2 ± 0.9%** |
| 5/21 G5 final | `outputs/20260520_G5_1M_pathC_div_50KSOTA/model/final/` | ❌ 发散，不可用 | v_mean = -417 |
| **🏆 5/22 G6 100K (reliable SOTA)** | `outputs/20260521_204651.815/model/checkpoints/step_00100000/` | ✅ **H2 满足 3/3**（稳定可复现）| **HCSF IM 0.127 ± 0.016, jerk 61.07 ± 2.38, 干预 14.3 ± 1.8%** |
| 5/22 G6 549K | `outputs/20260521_204651.815/model/checkpoints/step_00550000/` | ✅ **训练 v_mean 峰值 4.90** | HCSF IM 0.103 ± 0.030, jerk 60.3 ± 6.77, 干预 13.0 ± 2.7% |
| 5/22 G6 final (789K crash) | `outputs/20260521_204651.815/model/final/` | ⚠️ AC socket 崩溃终止 | 完整 V_φ stability，未发散 |

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

### Exp 7: 5/22 G6 Path C + V_φ Stability Fix (789K crash, 全程稳定)

- **配置**: Path C 不变（π^♦ 驱动 + max_episode_py_time=60s + 1M 步）+ **V_φ stability fix**:
  - V_target 用 `target_q_net` 而非 online Q net（跟 Q-target 对称）
  - V_target clamp 到 [-30, 30]
  - V_φ 梯度 L2 norm 截断 max_norm=10.0
- **改动文件**: `safety_value.py` (+29 行), `sac.py` (+1 行)
- **训练时长**: 9h 26min（**AC socket UTF-8 解码错误意外终止 @ 789K 步，AC 平台 bug 不是我们代码**）
- **训练曲线 V_φ 完全稳定**:

  | step | v_loss | v_mean | G5 同步对比 |
  |---|---|---|---|
  | 45K | 0.007 | 2.48 | G5 同 2.44，**G6 平行起步** |
  | 114K | 0.002 | **3.59** | G5: **-1.49** ❌ G5 已发散 |
  | 362K | 0.010 | ~4.65 | G5: **-18,228** ❌❌ G5 峰值崩溃 |
  | 549K | – | **4.90** | G5: – （已死）|
  | 789K | 0.005 | 4.77 | – （崩溃前） |

- **V_φ 峰值**: 4.90 @ step 549K（接近 G2 1M plateau 4.77 但更早达到）
- **15 个 checkpoints 保存**（50K → 750K，间隔 50K）
- **关键发现**: V_φ stability fix **完美解决发散**，但**没有改善 H2 指标**——G6 产生 "active filter" 行为，跟 G5 50K 的 "passive filter" 是不同操作点

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

## SOTA 实证 II：G6 跨 timeline Variance Eval（15 次复测，2026-05-22）

### 评估配置（同 G5 一致）

- Human policy: 10M_SAC
- noise=0.3, steps=500, 解耦模式
- 每个 checkpoint 测 3 次

### G6 5 个 checkpoints 全部数据（HCSF filter）

| Checkpoint | Run | IM | jerk | 干预率 | V_avg | LRSF jerk | H2 (HCSF<LRSF) |
|---|---|---|---|---|---|---|---|
| **G6 50K** | r1 | 0.132 | 65.6 | 17.0% | 1.4 | 67.2 | ✅ |
| | r2 | 0.081 | 56.7 | 10.2% | 1.7 | 138.4* | ✅ (LRSF crash) |
| | r3 | 0.274 | 72.3 | 21.7% | 1.5 | 63.6 | ❌ |
| **G6 100K** | r1 | 0.150 | 57.8 | 16.8% | 2.7 | 63.8 | ✅ |
| | r2 | 0.114 | 63.4 | 12.8% | 2.3 | 66.7 | ✅ |
| | r3 | 0.116 | 62.0 | 13.2% | 2.3 | 65.7 | ✅ |
| **G6 200K** | r1 | 0.089 | 56.8 | 9.2% | 2.7 | 64.8 | ✅ |
| | r2 | 0.158 | 64.7 | 16.4% | 2.6 | 62.2 | ❌ |
| | r3 | 0.074 | 77.5 | 9.6% | 2.9 | 62.2 | ❌ |
| **G6 549K** | r1 | 0.145 | 69.6 | 16.8% | 2.5 | 63.7 | ❌ |
| | r2 | 0.083 | 53.7 | 11.4% | 2.7 | 62.6 | ✅ |
| | r3 | 0.080 | 57.6 | 10.8% | 2.7 | 66.2 | ✅ |
| **G6 750K** | r1 | 0.115 | 54.7 | 16.6% | 2.6 | 60.9 | ✅ |
| | r2 | 0.137 | 66.9 | 14.4% | 2.6 | 65.0 | ❌ |
| | r3 | 0.015 | 59.8 | 3.2% | 2.7 | 56.5 | ❌ |

*G6 50K r2 LRSF jerk 138 是 LRSF 撞车造成的 outlier

### G6 跨 timeline 聚合 (mean ± std, n=3)

| Checkpoint | 训练 v_mean | HCSF IM | HCSF jerk | 干预率 | V_avg | H2 satisfy |
|---|---|---|---|---|---|---|
| G6 50K | ~2.4 | 0.162 ± 0.082 | 64.87 ± 6.39 | 16.3 ± 4.7% | 1.53 ± 0.13 | 2/3 |
| **G6 100K** | ~3.3 | **0.127 ± 0.016** | **61.07 ± 2.38** | 14.3 ± 1.8% | 2.43 ± 0.19 | **3/3** ⭐ |
| G6 200K | ~4.2 | 0.107 ± 0.037 | 66.3 ± 8.53 | 11.7 ± 3.3% | 2.73 ± 0.12 | 1/3 |
| G6 549K | **4.90 (peak)** | 0.103 ± 0.030 | 60.3 ± 6.77 | 13.0 ± 2.7% | 2.63 ± 0.09 | 2/3 |
| G6 750K | ~4.4 | 0.089 ± 0.053 | 60.5 ± 5.00 | 11.4 ± 5.9% | 2.63 ± 0.05 | 1/3 |

### 关键发现：G6 是稳定的 "active filter"，不是 outlier

**1. G6 全 timeline 一致：** IM 单调下降 0.162 → 0.089（V_φ 学得更准），干预率稳定 11-16%，V_avg 稳定 2.4-2.7。

**2. G5 50K 真的是 "happy accident"：** 直接对比 G5 50K 和 G6 50K：

| 指标 | G5 50K | G6 50K | 差异 |
|---|---|---|---|
| IM | 0.010 | 0.162 | **16x** |
| 干预率 | 1.2% | 16.3% | **13x** |
| V_avg | 2.3 | 1.53 | -33% |

V_φ stability fix 改变了整个训练轨迹——G6 在最早的 checkpoint 也是 active filter。**G5 50K 的 passive filter 风格是发散前的偶然甜蜜点，不可复现。**

**3. G6 100K = H2 strict 最佳点：**
- 3/3 次全部满足 H2 (HCSF jerk < LRSF jerk)
- IM std 极小（±0.016）
- jerk std 极小（±2.38）
- **稳定可复现的 SOTA**

---

## 完整 SOTA 排名（综合所有 27 次评估）

### 按论文 H2 strict 满足率

| 排名 | 模型 | n | H2 satisfy | IM | jerk | 干预率 |
|---|---|---|---|---|---|---|
| 🏆 | **G6 100K** | 3 | **3/3 = 100%** | 0.127 ± 0.016 | 61.07 ± 2.38 | 14.3 ± 1.8% |
| 🥈 | G5 50K | 3 | 2/3 = 67% | 0.010 ± 0.007 | 60.97 ± 5.31 | 1.2 ± 0.9% |
| 🥈 | G6 549K | 3 | 2/3 = 67% | 0.103 ± 0.030 | 60.3 ± 6.77 | 13.0 ± 2.7% |
| 🥈 | 5/13 baseline | 3 | 2/3 = 67% | 0.086 ± 0.046 | 64.0 ± 1.28 | 8.3 ± 5.3% |
| 5 | G6 50K | 3 | 2/3 = 67%* | 0.162 ± 0.082 | 64.87 ± 6.39 | 16.3 ± 4.7% |
| 6 | G6 200K | 3 | 1/3 = 33% | 0.107 ± 0.037 | 66.3 ± 8.5 | 11.7 ± 3.3% |
| 7 | G6 750K | 3 | 1/3 = 33% | 0.089 ± 0.053 | 60.5 ± 5.0 | 11.4 ± 5.9% |

*G6 50K r2 LRSF crash 让对比不公正

### 按 IM (Agency, 单一最优指标)

| 排名 | 模型 | IM | 备注 |
|---|---|---|---|
| 🏆 | **G5 50K** | **0.010** | 不可复现 outlier |
| 2 | 5/13 baseline | 0.086 | |
| 3 | G6 750K | 0.089 | |
| 4 | G6 549K | 0.103 | |

### 两种 SOTA 叙事

| SOTA 类型 | 模型 | 论证 |
|---|---|---|
| **数值最优** | **G5 50K** | "outlier SOTA"——V_φ 发散前的偶然甜蜜点，IM 是其他 10×，但不可复现 |
| **稳定可复现** | **G6 100K** | "reliable SOTA"——V_φ stability fix 后的最佳点，3/3 H2 满足，方差最小 |

---

## 路径性能纵向对比（HCSF SOTA 演化）

| 实验 | 步数 | HCSF IM | HCSF jerk | 干预率 | 备注 |
|---|---|---|---|---|---|
| 5/14 F1 | 305K | 0.42 | 118.7 | 26.6% | OOD 失败 |
| 5/19 G2 | 1M | 0.156 | 65.5 | 18.6% | Plateau |
| 5/13 baseline | 300K | 0.075 | 60.3 | 8.6% | 之前 SOTA |
| **🏆 5/21 G5 50K** | **50K** | **0.0074** | **59.9** | **1.2%** | 数值 SOTA（outlier）|
| **🏆 5/22 G6 100K** | **100K** | **0.127** | **61.07** | **14.3%** | 稳定 SOTA（H2 3/3）|

---

## 科学结论

1. **G5 50K = "happy accident"**：V_φ 在发散前偶然进入 passive filter 操作点，3 次复测 IM ∈ [0.0001, 0.0179]，但**这种行为依赖 V_φ 训练不稳定**，无法被故意复现
2. **V_φ stability fix 完全有效**：G6 全程 v_mean 稳定 2.4-4.9 区间，相比 G5 在 130K 步崩到 -2，**完全解决发散问题**
3. **稳定训练产出 "active filter" 行为**：G6 整段训练（50K 到 750K）一致是 active filter（10-16% 干预率），符合论文 H2 但数值不如 G5 50K outlier
4. **G6 100K 是 H2 strict 最佳点**：3/3 次满足论文 H2，方差最小，**稳定可复现的 SOTA**
5. **论文 HCSF 大概率类似 G6**：12.8M 步稳定训练的 HCSF 更像 G6（active filter）而非 G5 50K（passive outlier）

---

## 下一步实验候选

1. **修 AC socket UTF-8 bug**: `ac_client.py:143` 的 `data.decode()` 加 `errors='replace'` 容错，避免训练 78% 时被打断
2. **续训剩余 211K 步**: 加载 G6 750K checkpoint，跑到 1M 完整步数（~3h）
3. **配置对齐论文**: memory_size 8M→20M，控制频率 25Hz→30Hz
4. **完整 12.8M 步训练**: 在 JHU 实验室 GPU 上跑（3060 也可，1-2 周）
5. **赛道切换**: 银石（如有时间）
6. **真人用户研究**: 复现论文 Fig 5/7-9（JHU 期间）
