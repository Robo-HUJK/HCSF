# 研究背景：Safety with Agency

**论文标题：** "Safety with Agency: Human-Centered Safety Filter with Application to AI-Assisted Motorsports"（带能动性的安全：以人为中心的安全滤波器及其在 AI 辅助赛车中的应用）
**作者：** Donggeon David Oh*, Justin Lidard*, Haimin Hu, Himani Sinhmar, Elle Lazarski, Deepak Gopinath, Emily S. Sumner, Jonathan A. DeCastro, Guy Rosman, Naomi Ehrich Leonard, Jaime Fernández Fisac
**所属机构：** Princeton University, Toyota Research Institute
**ArXiv：** 2504.11717v4 [cs.RO], 2025 年 12 月 17 日

---

## 核心问题

传统的安全滤波器（如 Last-Resort Safety Filter / LRSF，"最后一刻"安全滤波器）只在系统抵达安全集边界时才进行突兀的干预，完全无视人类操作者的意图。这种行为会导致不连续的、急促的修正动作，破坏了人类的能动性（agency）与对系统的信任。本论文提出了 **以人为中心的安全滤波器（Human-Centered Safety Filter, HCSF）**，它在仍然保证安全的前提下，通过最小程度地偏离人类动作来保留人类的能动性。

---

## 核心方法

### 问题建模

离散时间非线性动力学：

```
x_{t+1} = f(x_t, u_t)                    (1)
```

失败集（failure set）由 Lipschitz 连续的安全裕度函数 g : X → ℝ 定义：

```
F := {x ∈ X | g(x) < 0}                  (2)
```

安全滤波器将任务策略的输出映射为安全动作：

```
u_t = φ(x_t, π^task)                      (3)
```

### Hamilton-Jacobi 安全值函数

安全 Bellman 方程（不动点）：

```
V(x) = min{ g(x),  max_{u∈U} V(f(x,u)) }         (4)
```

最大安全集：

```
S* := {x ∈ X | V(x) ≥ 0} ⊂ F^c                   (5)
```

状态-动作（Q-函数）变体——部署时无需系统动力学：

```
Q(x,u) = min{ g(x),  max_{u'∈U} Q(f(x,u), u') }   (6)
```

### 最后一刻安全滤波器（LRSF）——基线方法

当 V(x) = 0 时完全切换到回退策略 π^♦ := argmax_{u∈U} Q(x,u)：

```
u(x) = { π^task(x)   若 x ∈ S*  且  V(x) > 0
        { π^♦(x)      否则                       (7)
```

### 离散时间控制屏障函数（DCBF）

若 S = {x | h(x) ≥ 0} ⊂ F^c 且 ∃α ∈ (0,1]，则函数 h : X → ℝ 是一个 DCBF：

```
sup_{u∈U} Δh(x,u) ≥ −αh(x),  ∀x ∈ X             (8)
其中 Δh(x,u) := h(f(x,u)) − h(x)
```

基于 DCBF 的滤波器通过求解优化问题（而非硬切换）来工作：

```
u(x) = argmin_{u∈U}  ||u^task(x) − u||²           (9a)
        s.t.  Δh(x,u) ≥ −αh(x)                    (9b)
```

### 关键命题：V(·) 是有效的 Q-CBF（命题 1）

安全值函数 V(x)（公式 4 的不动点解）是一个有效的 DCBF。对应的 **Q-CBF 约束**（无模型，无需动力学）：

```
Q(x, u) ≥ γV(x),   γ ∈ [0, 1)                    (10)
```

它完全替代了依赖动力学的 DCBF 约束（9b）。

### HCSF 定义（定义 2）

在每个时间步求解一个最优控制问题（OCP），以满足 Q-CBF 的同时最小程度地修改人类动作：

```
u*(x) = argmin_{u∈U}  ||u^human(x) − u||²          (11a)
          s.t.  Q(x, u) ≥ γV(x)                    (11b)
```

γ ∈ [0,1) 控制 V(·) 每一步允许下降的速度。论文实验中 γ = 0.7 为最优值（在能动性与舒适性之间取得平衡）。

**命题 2（递归可行性）：** 对所有 γ ∈ [0,1) 和任意初始状态 x ∈ S*，问题 (11) 是递归可行的。

---

## 训练公式

### 时间折扣安全 Bellman 方程

为支持基于 RL 的近似：

```
V_φ(x) = (1 − γ_ENV)g(x) + γ_ENV min{ g(x), max_u Q_φ(x,u) }    (21)
```

### Q-网络损失（Bellman 残差）

```
L(φ) = E^{B,π}[ (Q_φ(x,u) − y)² ]                 (22)
其中 y := (1 − γ_ENV)g' + γ_ENV min{ g', Q_{φ'}(x', u') }
```

### 策略（回退策略 π^♦）梯度

```
L(θ) = E^{B,π}[ −Q(x,u) + α log π_θ(u|x) ]        (23)
```

（SAC 风格的熵正则化目标函数）

### HCSF 执行：求解 OCP（公式 11）

运行时，在控制空间中、于连接 u^human(x) 与 u^♦(x) 的线段上采样 **2000 个候选动作**，然后在满足 Q(x,u) ≥ γV(x) 的候选中选择使 ‖u^human − u‖² 最小的一个。

---

## 关键算法

### 算法 1：HCSF 训练

```
初始化策略参数 θ、值函数参数 φ、经验回放缓冲区 B
设置目标软更新系数 τ、折扣因子 γ_ENV、熵温度 α
for 每个训练 episode:
    观察初始状态 x_0
    for 每个环境步:
        u_t ~ π_θ(u_t | x_t)
        执行 u_t，观察安全裕度 g_t 和下一状态 x_{t+1}
        将 (x_t, u_t, g_t, x_{t+1}) 存入 B
    for i = 1..N_UTD:
        从 B 采样 minibatch (x_t, u_t, g_t, x_{t+1})
        计算目标值 y_t = (1−γ_ENV)g_t + γ_ENV min{g_t, Q_φ(x', u')}
        使用公式 (22) 更新 Q-函数
        使用公式 (23) 更新策略
        软更新目标 Q：φ ← τφ + (1−τ)φ
```

### 算法 2：HCSF 执行

```
for 每个环境步:
    观察当前状态 x
    观察人类动作 u^human(x)，计算回退策略动作 u^♦(x)
    在连接 u^human(x) 与 u^♦(x) 的线段上采样候选动作
    选择满足 (11b) 且使 (11a) 最小的动作 u
    将 u 施加到系统
```

---

## 实验设置

### 仿真器与硬件（§V-B, §V-C）

| 组件 | 详情 |
|-----------|---------|
| 仿真器 | Assetto Corsa（AC），高保真度，**黑盒动力学** |
| 赛道 | Silverstone Circuit（GP 布局） |
| 自车 | BMW Z4 GT3 |
| 对手车辆 | Mazda MX-5 ND（更弱性能，鼓励超车）50% 强度，30% 攻击性 |
| 训练 vs 用户研究对手数 | **训练用多个对手**，**用户研究用单个对手**（§V-C 关键差异）|
| 天气条件 | Weather: ideal, Track: optimum, 温度 26°C, 风速 0 km/h |
| 辅助 | Traction Control + Stability Control + ABS **开**；油耗/胎损 **关** |
| 方向盘/踏板 | Fanatec CSL DD QR2 + ClubSport GT Alcantara V2 + Clubsport V3 |
| 显示设备 | Samsung S39C FHD 75Hz Curved Monitor + Trak RS6 模拟器架 |
| 控制频率 | 30 Hz（SCI 控制循环） |
| 渲染频率 | 300 Hz（第一人称视角） |

### 训练细节（§V-E）

| 超参数 | 值 |
|----------------|-------|
| 训练设备 | RTX 4090 + AMD Ryzen 9 7950X 16-core |
| 总训练时间 | ~3 周（**12.8M 环境步**） |
| Replay buffer 容量 | **20M transitions** |
| 网络架构 | 3 层 MLP，每隐层 256 个神经元（Q 和 π^♦ 同结构） |
| **Batch size** | **128**（论文统一值，非 256） |
| 学习率 η | 3×10⁻⁴ |
| 折扣因子 γ_ENV | 0.992 |
| 目标软更新系数 τ | 0.005 |
| 熵温度 α | 可学习 |
| 优化器 | Adam |
| N_UTD（每环境步的梯度更新次数） | 1（actor 和 critic 各更新一次）|
| Q-CBF γ | 0.7（论文最优经验值，§IV + Appendix C-5） |

### 观测空间（133 维）

主要组件（除标注外均堆叠过去 4 个时间步）：
- 自车速度、距参考路径的间距、力反馈、RPM、加速度（×2）、档位、角速度、本地速度（×2）、侧滑角（×4）
- 距赛道边界的距离（×11）、是否出界标志
- 前瞻曲率（×12，1 个时间步）
- 控制输入（×3）、自车朝向、自车偏航角
- 对手：距离、方向、速度、朝向、偏航角、刹车状态

### 动作空间

```
U = [−1, 1]³   （转向、油门、刹车——增量式）
```

档位由 AC 自动处理。动作表示相对上一时刻控制量的增量（缓解信号振荡/抖动）。

### 安全裕度函数（§V-C-3）

```
g(x) = min(到赛道边界的有符号距离, 到最近对手的有符号距离)
F = {x | g(x) < 0}   （出界或发生碰撞）
```

### Episode 终止条件（§V-C-4，重要）

> "If the margin function becomes negative **or** if the vehicle remains stationary for an elongated time period, the episode terminates and the vehicle is **automatically reset to the closest point on the reference path**."

**只有两种自然终止：**
1. `g(x) < 0`（出界或碰撞）
2. 车长时间静止
3. **论文无人为 step cap**——episode 长度自然由 π^♦ 的失败频率决定

这个条件**同时应用于 neural synthesis 训练阶段和用户研究**。

### 训练流程（多阶段，§V-D + Appendix C）

> AC 重置时把车放在 reference path 上**静止**——直接训练会缺乏 near-failure 样本。
> 解决方案：两阶段 pipeline，先加速到 race 速度，再系统性推入危险状态。

1. **预热阶段（Warmup Phase）：** 用 "performance-oriented policy" 把车加速到现实速度。论文 §V-D 没明确指出 nominal/overtaking 策略组合概率（在 Appendix C-1 Table V）。我们的 config 使用：
   - 名义策略 / 超车策略 以 0.6/0.4 概率混合
   - 靠近对手 (P^warmup_oppo = 0.25) 或重刹车 (P^warmup_brake_eps = 0.4) 时提前终止
2. **初始化阶段（Initialization Phase）：** 系统性把车推入"易失败状态"，包括 **adversarial 和 random maneuvers near the boundary of the safe set**。
   - 对抗 (P^init_adv = 0.3) / 随机 (P^init_rand = 0.3) / 混合 (P^init_mix = 0.4)
   - Q 值低于 Q^init_term = 2 时按 P_term = 0.2 概率终止
3. **训练阶段（Training Phase）：** 标准 SAC（π^♦ 探索 + Q/V_φ 学习）。论文未明确 TRAINING 阶段 driver 是 π^♦ 还是 performance policy（隐含 π^♦——见 5/20 G5 实验验证）

### 视觉提示（§V-F）

- 屏幕上**纵向 + 横向箭头**显示 AI 修正方向和幅度
  - 箭头方向 = 偏左/右转、增/减油门
  - 箭头长度 ∝ 修正幅度 (‖u* − u^human‖)
- 每个箭头映射到独立的控制通道
- 不用音频提示（高速场景会增加认知负载）
- 同时给所有 participants 显示 **彩色 reference path**：
  - 绿色箭头 = 加速
  - 红色箭头 = 减速
- 目的：低带宽视觉提示促进透明协作 + 让非赛车手知道哪里加减速

---

## 用户研究

### 实验设计

| | HCSF | LRSF | None |
|--|------|------|------|
| 参与者人数 | 29 | 29 | 25 |
| 平均初始技能水平 | 2.17±0.54 | 2.21±0.86 | 2.28±0.68 |

**Session 结构：**
- Session 1：5 分钟，无辅助（基线）
- Session 2：10 分钟，分配滤波器（HCSF / LRSF / 安慰剂）
- Session 3：5 分钟，无辅助（过度依赖性检测）

### 评估指标

**定量轨迹指标：**
- **鲁棒性（Robustness）：** 每分钟出界次数（OOT/min）、每分钟碰撞次数（<3m 距离）、每分钟失败次数
- **能动性（Agency, IM）：** `IM(t) = ||u_t^human − φ(x_t, u_t^human)||₂`（公式 12）
- **舒适性（Comfort, jerk）：** `jerk(t) = ||p̈_t||₂`（公式 13）
- **舒适性（Comfort, ID）：** `ID(t) = ||φ(x_t, u_t^human) − φ(x_{t-1}, u_{t-1}^human)||₂²`（公式 14）

**定性指标（5 级李克特量表，肯定+否定两种问法）：** 鲁棒性、能动性、舒适性、满意度、可信度（Trustworthiness）、可预测性（Predictability）、可解释性（Interpretability）、能力感（Competence）。

**统计分析：** Mixed ANOVA（session × group）→ SME → Tukey's HSD。显著性标记：* p<0.05，** p<0.01，*** p<0.001。

---

## 关键结果

### 鲁棒性（H1 得到验证）
- HCSF：Session 2 中出界 (0.00/min) 和失败 (0.02/min) 接近零
- 显著优于无辅助组（p<0.001）
- 与 LRSF 相比无统计显著差异

### 能动性（H2 得到验证）
- HCSF 的干预频率比 LRSF 高（30.3% vs. 19.7% 的时间步）
- 但 HCSF 的干预幅度远小于 LRSF：**平均 IM = 0.184 vs. 0.305**
- HCSF 能动性评分与无辅助组无显著差异（p>0.05）
- 相比无辅助和 HCSF，LRSF 显著削弱了人类能动性

### 舒适性（H2 得到验证）
- HCSF 的平均 jerk 显著低于 LRSF（p<0.001）
- HCSF 的平均 I.D. 显著低于 LRSF（p<0.001）
- HCSF 舒适性评分与无辅助组无显著差异

### 满意度（H1 与 H2 均得到验证）
- Session 2 中 HCSF 的满意度显著高于 LRSF 和无辅助两组

### 赛车性能（圈速，Session 2）
| 组别 | Session 1 | Session 2 | Session 3 |
|-------|-----------|-----------|-----------|
| LRSF  | 3.74 min  | 3.08 min  | 3.27 min  |
| HCSF  | 3.59 min  | **2.99 min** | 3.21 min  |
| None  | 3.62 min  | 3.22 min  | 3.09 min  |

HCSF 是 Session 2 中唯一进入 3 分钟以内的组别（虽然不具统计显著性）。

### 滤波器专属指标
- 相比无辅助，HCSF 的 **可信度** 和 **能力感** 显著更高
- 相比无辅助，LRSF 显著更"难以预测"
- HCSF 的可解释性优于 LRSF

### 过度依赖风险
HCSF 组 Session 3 的圈速略慢于无辅助组（无统计显著性），暗示移除滤波器辅助后可能存在过度依赖问题。

---

## 局限性

1. HCSF 不直接优化赛车性能（如圈速）——只优化"安全 + 最小偏离"
2. 响应式设计仅处理即时威胁，缺乏前瞻性引导
3. 长期使用可能导致过度依赖，妨碍无辅助驾驶技能的发展
4. 对手模型非对抗性，碰撞鲁棒性未完全保证
5. γ = 0.7 是经验选择；未来工作可以使用 Parametric CBFs 来联合优化 γ

---

## 当前复现状态（截至 2026-05-21）

**总览**：Phase 0-5 全部实现，端到端 pipeline 可运行。5/21 产出 G5 50K SOTA 模型（解耦评估满足论文 H2 假设）。详细实验结果见 [`experiments.md`](experiments.md)，每日工作记录见 [`daily_log.md`](daily_log.md)。

| 阶段 | 完成度 | 产出 |
|---|---|---|
| Phase 0 基础设施 | ✅ 100% | Git + W&B + 基线 SAC |
| Phase 1 g(x) 训练循环 | ✅ 100% | `ac_env.py` / `replay_buffer.py` / `sac.py` 全链路 |
| Phase 2 安全值函数 V_φ | ✅ 100% | `safety_value.py` 实现 Eq. 21 |
| Phase 3 滤波器 | ✅ 100% | `hcsf_filter.py` + `lrsf_filter.py`（候选 500 vs 论文 2000）|
| Phase 4 训练课程 | ⚠️ 90% | `training_phases.py` 含 warmup + INIT；warmup policy 用 10M_SAC 代理（未实现 Eq. 17-20 训练）|
| Phase 5 评估 | ✅ 100% | `evaluate_hcsf.py` + 解耦评估 + IM/jerk/ID 指标 |
| 对手集成 | ⚠️ 部分 | g(x) 加对手距离已实现，但 Race 模式 OOD 训练失败（5/14）|
| 用户研究（83 真人）| ❌ 未做 | 需 JHU 期间实体设备（7 月第二周到）|
| 视觉提示（§V-F）| ❌ 未做 | 需 AC 插件层渲染 |
| 12.8M 步完整训练 | ❌ 未做 | 当前最长 1M 步；3060 上 1-2 周可行 |

---

## 论文组件 → 代码文件对应表

> 状态图例：✅ 已实现且与论文一致 | ⚠️ 部分实现/有差异 | ❌ 缺失 | 🆕 我们的新增

### 环境与观测（§V-B, §V-C）

| 论文组件 | 论文出处 | 代码文件 | 状态 | 备注 |
|---|---|---|---|---|
| Gym 环境接口 | §V-B | `AssettoCorsaEnv/ac_env.py` | ✅ | |
| 133 维观测向量 | App. B, Table IV | `ac_env.py` | ⚠️ | 我们 125 维（部分可选位关闭，如 `enable_task_id_in_obs`）|
| 动作 U=[−1,1]³ 增量式 | §V-C-2 | `ac_env.py: use_relative_actions` | ✅ | |
| 距参考路径间距 d_gap | §V-C-3 | `gap_cpu.py` / `gap_torch.py` | ✅ | |
| 射线投射 ×11 ×4 timesteps | App. B | `sensors_ray_casting.py` | ✅ | |
| 前瞻曲率 ×12 | App. B | `ac_env.py` + `reference_lap.py` | ✅ | |
| 对手相对位置/速度/朝向/刹车 | App. B | `ac_env.py` + `ac_client.py`（5/14 加 brakeStatus）| ✅ | |
| g(x) = min(赛道, 对手) | Eq. 2, §V-C-3 | `ac_env.py` | ✅ | 实际 `min(track, opp_signed - 5)`，加 5m buffer |
| g(x)<0 自然终止（无 step cap）| §V-C-4 | `ac_env.py: enable_out_of_track_termination` | ✅ | G5 用 `max_episode_py_time=60s` 短 episode hack 偏离论文 |
| 静止过久终止 | §V-C-4 | `ac_env.py: enable_low_speed_termination` | ✅ | |
| Reset 到 reference path 最近点 | §V-C-4 | AC plugin 内置 | ⚠️ | 实现细节未公开（Q1 待 David 确认）|
| AC socket 30 Hz 控制 | §V-B | `ac_client.py` | ⚠️ | 我们 25 Hz（`ego_sampling_freq: 25`）|

### 神经网络（§V-A, §V-E）

| 论文组件 | 论文出处 | 代码文件 | 状态 | 备注 |
|---|---|---|---|---|
| 双 Q-网络 Q_φ(x,u) | Eq. 22 | `discor/network.py: TwinnedStateActionFunction` | ✅ | |
| 3 层 MLP, 256 units | §V-E | `config.yml: q_hidden_units = [256,256,256]` | ✅ | |
| 最佳努力回退策略 π^♦(x) | Eq. 7, §V-A | `network.py: GaussianPolicy` | ✅ | SAC policy 在 HCSF 框架下即 π^♦（§V-A 第 76-78 行确认）|
| 安全值函数 V_φ(x) | Eq. 4, 21 | `algorithm/hcsf/safety_value.py` | ✅ | 5/12 实现 |

### SAC 训练（算法 1, §V-A）

| 论文组件 | 论文出处 | 代码文件 | 状态 | 备注 |
|---|---|---|---|---|
| HCSF Q-target: `(1−γ)g + γ·min{g, Q'}` | Eq. 21, 22 | `sac.py:194-204` | ✅ | 5/12 实现 |
| 策略损失 max Q（含 entropy 项）| Eq. 23 | `sac.py:90-135` | ✅ | SAC 风格熵正则 |
| Replay buffer 存 (x, u, g, x') | 算法 1 第 9 行 | `replay_buffer.py` | ✅ | 5/12 加 g 字段 |
| 目标 Q 软更新 | 算法 1 第 16 行 | `sac.py` / `agent.py` | ✅ | τ=0.005 |
| 熵温度 α 可学习 | §V-E | `sac.py: log_alpha` | ✅ | |
| N_UTD=1 每环境步一次梯度 | §V-E | `agent.py` | ✅ | |
| V_φ 训练（Eq. 21 Bellman）| Eq. 21 | `safety_value.py: SafetyValueTrainer` | ✅ | 跟 Q 同步更新 |
| V_φ target clipping / early stopping | (论文未明示) | — | ❌ | G5 1M 步出现 V_φ 发散——可能论文 12.8M 步足够长不触发，或有未公开的稳定技巧 |

### HCSF 滤波器（§IV, 算法 2）

| 论文组件 | 论文出处 | 代码文件 | 状态 | 备注 |
|---|---|---|---|---|
| Q-CBF 约束: Q(x,u) ≥ γV(x) | Prop. 1, Eq. 10 | `hcsf_filter.py` | ✅ | |
| HCSF OCP: argmin‖u^h−u‖² s.t. Q≥γV | Def. 2, Eq. 11 | `hcsf_filter.py` | ✅ | |
| 候选动作采样 | 算法 2, App. C | `hcsf_filter.py` | ⚠️ | 我们 **500** 候选，论文 **2000** |
| γ_CBF = 0.7 | §IV, App. C-5 | `hcsf_filter.py` 硬编码 | ✅ | |
| LRSF 基线（硬切换 π^♦）| Eq. 7 | `lrsf_filter.py` | ✅ | |
| 视觉提示（红绿箭头）| §V-F | — | ❌ | 未实现 |

### 训练课程（§V-D, App. C-1）

| 论文组件 | 论文出处 | 代码文件 | 状态 | 备注 |
|---|---|---|---|---|
| Warmup phase（performance-oriented policy 加速）| §V-D | `training_phases.py: _warmup_action` | ⚠️ | 用 10M_SAC 代理，**非 Eq. 17-20 训练的策略** |
| Nominal/Overtaking 0.6/0.4 混合 | App. C-1 Table V | `config.yml: warmup_cfg.P_over` | ⚠️ | 当前 P_over=0（仅 nominal）|
| Nominal warmup 训练（Eq. 17）| App. C-2 | — | ❌ | 未实现（Q2 待 David 确认 frozen 是否需要单独训）|
| Overtaking warmup 训练（Eq. 18-20）| App. C-2 | — | ❌ | 同上 |
| Init phase（对抗/随机/混合）| §V-D, App. C-1 | `training_phases.py: _init_action` | ✅ | adversarial 用 argmin Q（500 候选）|
| P_adv=0.3, P_rand=0.3, P_mix=0.4 | Table V | `config.yml: init_cfg` | ✅ | |
| P_FT（满油门）=0.4 | Table V | `config.yml: init_cfg.P_FT` | ✅ | |
| Q^init_term=2 | Table V | `config.yml: init_cfg.Q_init_term` | ✅ | |
| P_term=0.2 | Table V | `config.yml: init_cfg.P_term` | ✅ | |
| 对手邻近提前终止 | App. C-1 | `training_phases.py`（stub）| ⚠️ | 代码 stub 在，未启用 |
| 重刹车提前终止 | App. C-1 | `training_phases.py:204-208`（注释）| ⚠️ | 注释掉了 |
| **T_init_max=3s 硬上限** | (论文未明示) | `config.yml: init_cfg.T_init_max` | 🆕 | 我们的新增（防 INIT 占满 episode，5/20 G3 教训）|

### 评估（§VI）

| 论文组件 | 论文出处 | 代码文件 | 状态 | 备注 |
|---|---|---|---|---|
| IM 指标 ‖u^h − φ(x, u^h)‖₂ | Eq. 12 | `evaluate_hcsf.py: MetricsTracker.step` | ✅ | |
| Jerk 指标 ‖p̈‖₂ | Eq. 13 | `evaluate_hcsf.py` | ✅ | |
| ID 指标 ‖φ_t − φ_{t-1}‖₂² | Eq. 14 | `evaluate_hcsf.py` | ✅ | |
| HCSF/LRSF/None 三组对比 | §VI-A | `evaluate_hcsf.py: run_trial` | ✅ | |
| **解耦评估** (u^human + filter 独立)| §VI / App. C-4 ablation | `evaluate_hcsf.py: --human-model/--filter-model` | ✅ | 5/18 工程化实现 |
| **Q-CBF margin** = Q(x, π^♦) − γV | (论文未导出指标)| `evaluate_hcsf.py: qcbf_slack` | 🆕 | 我们的诊断指标 |
| 用户研究（83 真人 × 3 session）| §VI-C | — | ❌ | JHU 期间做 |

### 配置参数（§V-E Table VI）

| 论文组件 | 论文值 | 当前 `config.yml` | 状态 |
|---|---|---|---|
| γ_ENV | 0.992 | `SAC.gamma: 0.992` | ✅ |
| τ | 0.005 | `SAC.target_update_coef: 0.005` | ✅ |
| **Batch size** | **128** | `Agent.batch_size: 128` | ✅ **对齐论文** |
| 学习率 η | 3e-4 | `SAC.q_lr / policy_lr / entropy_lr: 0.0003` | ✅ |
| **Replay buffer** | **20M** | `Agent.memory_size: 8M` | ⚠️ 我们 8M（RAM 受限）|
| γ_CBF | 0.7 | `hcsf_filter.py` 内 | ✅ |
| 控制频率 | 30 Hz | `AssettoCorsa.ego_sampling_freq: 25` | ⚠️ 我们 25 Hz |
| 训练步数 | 12.8M | 最多 1M | ❌ 算力时间限制 |

---

## 与论文的关键差异（写 reproduction report 时必须诚实标注）

### 1. 训练规模（受 3060 + 时间预算限制）

| 项 | 论文 | 当前 |
|---|---|---|
| 训练步数 | 12.8M（3 周 RTX 4090）| 1M（12h RTX 3060）|
| Replay buffer | 20M | 8M |
| 控制频率 | 30 Hz | 25 Hz |

**影响**：V_φ 学习不充分，但 SOTA G5 50K 已展示方法有效。JHU 期间跑 12.8M。

### 2. Warmup 策略

| 项 | 论文 | 当前 |
|---|---|---|
| Nominal warmup policy | Eq. 17 奖励训练 | 用 **Remonda et al. 2024 的 10M_SAC** 代理 |
| Overtaking warmup policy | Eq. 18-20 训练 | 未实现 |
| Warmup driver 混合 | nominal 60% / overtaking 40% | 仅 nominal（P_over=0）|

**待 David 确认**（暑期三方会议）：tentatively frozen 是否需要按 Eq. 17-20 单独训练？我们的 10M_SAC 代理够用吗？

### 3. 训练场景

| 项 | 论文 | 当前 |
|---|---|---|
| 赛道 | Silverstone GP | **Barcelona GP**（AC 默认）|
| 训练对手数 | 多个 | 0 个（Hotlap SOTA）或 1 个（5/14 失败案例）|
| 用户研究对手数 | 1 个 | N/A |

**影响**：Hotlap 无对手 → g(x) 始终 > 0 → V_φ Bellman target 可能退化（5/19 G2 plateau 印证）。短 episode hack 部分补偿。

### 4. HCSF 候选采样

| 项 | 论文 | 当前 |
|---|---|---|
| 候选数量 | 2000 | **500** |
| 采样区间 | u^h → u^♦ 线段 | 同 |

**影响**：候选少 → 数值上稍多 fallback。SOTA 仅 2 次/500 步，可接受。

### 5. Reset 实现（Q1 待 David 暑期当面确认）

- 论文：reset 到 reference path 最近点
- 我们：AC plugin 内置 reset，**细节未公开**

### 6. 用户研究 + 视觉提示

- 视觉提示（§V-F 红绿箭头）：未实现
- 83 人用户研究：未做（待 JHU 期间）

### 7. 我们的新增（不在论文里）

- **`T_init_max=3s` 硬上限**：防 INIT 占满 episode（5/20 G3 教训）
- **`max_episode_py_time=60s` 短 episode hack**（5/20 G5 Path C）：1M 步预算下补偿 π^♦ 失败频率不足
- **解耦评估方法学**（5/18）：filter policy 不会开车时单模型评估 spurious
- **Q-CBF margin metric**（`qcbf_slack`）：诊断 V_φ 过激进
- **V_φ 发散现象 + Fix**（5/22 G6 解决）：G5 1M 在 100K-130K 步训练发散，v_mean 从 2.44 → -18,228（论文未报告）。**已落地 fix**（`safety_value.py`）:
  1. V_target 用 `target_q_net`（跟 Q-target 对称稳定 bootstrap），不用 online Q
  2. V_target clamp 到 [-30, 30] 防极端值
  3. V_φ gradient L2 norm clip max_norm=10
  - **验证**: G6 全 789K 步 V_φ 在 [3.5, 4.9] 健康区间，无发散
- **两种 SOTA 操作点发现**（5/22）:
  - "Outlier SOTA"（G5 50K）: 1.2% 干预率，IM 0.010（**V_φ 发散前偶然甜蜜点，不可复现**）
  - "Reliable SOTA"（G6 100K）: 14.3% 干预率，IM 0.127，**H2 satisfy 3/3 全候选最佳**
  - 揭示**稳定训练自然产出 "active filter" 行为**——可能更接近论文 12.8M 训练的真实输出