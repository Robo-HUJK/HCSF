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

### 仿真器与硬件

| 组件 | 详情 |
|-----------|---------|
| 仿真器 | Assetto Corsa（AC），高保真度，黑盒动力学 |
| 赛道 | Silverstone Circuit（GP 布局） |
| 自车 | BMW Z4 GT3 |
| 对手车辆 | Mazda MX-5 ND（50% 强度，30% 攻击性） |
| 方向盘/踏板 | Fanatec CSL DD QR2 + ClubSport GT Alcantara V2 + Clubsport V3 |
| 显示设备 | Samsung S39C FHD 75Hz Curved Monitor + Trak RS6 模拟器架 |
| 控制频率 | 30 Hz |
| 渲染频率 | 300 Hz（第一人称视角） |

### 训练细节

| 超参数 | 值 |
|----------------|-------|
| 训练设备 | RTX 4090 + AMD Ryzen 9 7950X 16-core |
| 总训练时间 | ~3 周（1280 万环境步） |
| Replay buffer 容量 | 2000 万 transitions |
| 网络架构 | 3 层 MLP，每隐层 256 个神经元 |
| Batch size | 128（Q/policy 更新），256（SAC） |
| 学习率 η | 3×10⁻⁴ |
| 折扣因子 γ_ENV | 0.992 |
| 目标软更新系数 τ | 0.005 |
| 熵温度 α | 可学习 |
| N_UTD（每环境步的梯度更新次数） | 1 |

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

### 安全裕度函数

```
g(x) = min(到赛道边界的有符号距离, 到最近对手的有符号距离)
F = {x | g(x) < 0}   （出界或发生碰撞）
```

### 训练流程（多阶段）

1. **预热阶段（Warmup Phase）：** 通过名义策略（nominal）/超车策略（overtaking）以 0.6/0.4 的概率组合，将自车加速到现实速度。靠近对手（P^warmup_oppo = 0.25）或重刹车（P^warmup_brake_eps = 0.4）时提前终止。
2. **初始化阶段（Initialization Phase）：** 系统性地将自车推入易失败状态，使用对抗（P^init_adv = 0.3）、随机（P^init_rand = 0.3）或混合（P^init_mix = 0.4）初始化策略。当 Q 值低于 Q^init_term = 2 时终止。

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

## 与本仓库的关系

本论文直接驱动并实现于本 Assetto Corsa Gym 仓库：
- AC Gym 环境（`assetto_corsa_gym/AssettoCorsaEnv/ac_env.py`）提供 HCSF 训练所用的 Gym 接口
- `algorithm/discor/algorithm/sac.py` 中的 SAC 算法对应于 Q-函数 / 回退策略的训练（算法 1，公式 22–23）
- 133 维观测向量、安全裕度函数、Silverstone / BMW Z4 GT3 的设置与论文 ODD（Appendix B, §V-C）一致
- 预热 / 初始化训练流水线（Appendix C）即上文描述的多阶段方法
- HCSF 执行（算法 2）在部署时作为安全过滤层，叠加在人类方向盘输入之上

---

## 论文组件 → 代码文件对应表

> 状态图例：✅ 已实现且与论文一致 | ⚠️ 部分实现 | ❌ 缺失

### 环境与观测

| 论文组件 | 论文出处 | 代码文件 | 关键行号 | 状态 |
|----------------|----------------|-----------|-----------|--------|
| Gym 环境接口 | §V-B | `assetto_corsa_gym/AssettoCorsaEnv/ac_env.py` | `step()`, `reset()`, `__init__()` | ✅ |
| 133 维观测向量 | Appendix B, Table IV | `ac_env.py` | ~337–355 | ✅ |
| 动作空间 U = [−1,1]³（增量式） | §V-C-2 | `ac_env.py` | 动作空间定义 | ✅ |
| 距参考路径间距（d_gap） | §V-C-3 | `AssettoCorsaEnv/gap_cpu.py`, `gap_torch.py` | `get_gap()` | ✅ |
| 射线投射距离（赛道边界 ×11） | Appendix B | `AssettoCorsaEnv/sensors_ray_casting.py` | — | ✅ |
| 前瞻曲率（×12） | Appendix B | `ac_env.py` + `reference_lap.py` | — | ✅ |
| 对手相对位置/速度/朝向 | Appendix B | `ac_env.py` | 通过 AC 遥测部分获取 | ⚠️ |
| 安全裕度 g(x) = min(赛道距离, 对手距离) | Eq. 2, §V-C-3 | `ac_env.py` | 仅有 OOT 标志；**缺少对手距离** | ⚠️ |
| g(x) < 0 时 episode 终止 | §V-C-4 | `ac_env.py` | ~534–584 | ⚠️ 无软着陆 |
| AC socket 客户端（30 Hz 控制） | §V-B | `AssettoCorsaEnv/ac_client.py` | — | ✅ |

### 神经网络

| 论文组件 | 论文出处 | 代码文件 | 关键行号 | 状态 |
|----------------|----------------|-----------|-----------|--------|
| 双 Q-网络 Q_φ(x,u) | §V-A, Eq. 22 | `algorithm/discor/network.py` | `TwinnedStateActionFunction` | ✅ |
| 3 层 MLP，每层 256 单元 | §V-E | `network.py` | hidden_units 参数 | ✅ |
| 最佳努力回退策略 π^♦(x) | Eq. 7, §V-A | `network.py` | `StateIndependentPolicy` | ⚠️ 作为 SAC 策略 π 训练，未单独以 argmax_u Q 形式训练 |
| 安全值函数 V(x) | Eq. 4–5 | — | **不存在** | ❌ |

### SAC 训练（算法 1）

| 论文组件 | 论文出处 | 代码文件 | 关键行号 | 状态 |
|----------------|----------------|-----------|-----------|--------|
| Q 损失：Bellman 残差（公式 22） | Eq. 22, §V-A | `algorithm/discor/algorithm/sac.py` | `calc_q_loss()` ~170–206 | ✅ |
| 策略损失：熵正则化（公式 23） | Eq. 23 | `sac.py` | `update_policy_and_entropy()` ~90–135 | ✅ |
| 时间折扣 target：(1−γ)g + γ min{g, Q} | Eq. 21 | `sac.py` | target_qs 计算 | ❌ **target 中无 g(x)** |
| Replay buffer B 存储 (x,u,g,x') | 算法 1 第 9 行 | `algorithm/discor/replay_buffer.py` | `ReplayBuffer` | ⚠️ **未存储 g** |
| 目标 Q 软更新 φ←τφ+(1−τ)φ | 算法 1 第 16 行 | `sac.py` / `agent.py` | target 网络更新 | ✅ |
| 熵温度 α（可学习） | §V-E | `sac.py` | `log_alpha` | ✅ |
| N_UTD = 1 每环境步一次梯度更新 | §V-E, Table VI | `agent.py` | 训练循环 | ✅ |

### HCSF 特有组件（核心贡献——全部缺失）

| 论文组件 | 论文出处 | 代码文件 | 关键行号 | 状态 |
|----------------|----------------|-----------|-----------|--------|
| Q-CBF 约束：Q(x,u) ≥ γV(x) | Prop. 1, Eq. 10 | — | **不存在** | ❌ |
| HCSF OCP（公式 11）：argmin ‖u^h−u‖² s.t. Q≥γV | Def. 2, Eq. 11 | — | **不存在** | ❌ |
| 候选动作采样（u^h→u^♦ 线段上 2000 个点） | 算法 2, Appendix C-4 | — | **不存在** | ❌ |
| γ = 0.7 设计参数 | §IV, Appendix C-5 | `config.yml` | 未配置 | ❌ |
| LRSF 基线（V=0 时硬切换，公式 7） | Eq. 7 | — | **不存在** | ❌ |
| 视觉提示（箭头长度 ∝ ‖u*−u^h‖） | §V-F | — | **不存在** | ❌ |

### 训练课程（Appendix C——全部缺失）

| 论文组件 | 论文出处 | 代码文件 | 关键行号 | 状态 |
|----------------|----------------|-----------|-----------|--------|
| 预热阶段（名义 + 超车策略） | App. C-1 | `agent.py` | 仅有标准随机探索 | ❌ |
| 名义预热策略 π^nom_over（基于公式 17 训练） | App. C-2, Eq. 17 | — | **不存在** | ❌ |
| 超车预热策略（基于公式 18–20 训练） | App. C-2, Eqs. 18–20 | — | **不存在** | ❌ |
| 初始化阶段（对抗/随机/混合） | App. C-1 | `agent.py` | **不存在** | ❌ |
| 对手邻近提前终止 | App. C-1 | `ac_env.py` | **不存在** | ❌ |
| 重刹车提前终止 | App. C-1 | `ac_env.py` | **不存在** | ❌ |
| Q 值阈值终止（Q^init_term = 2） | App. C-1 | — | **不存在** | ❌ |

### 配置参数

| 论文组件 | 论文出处 | 配置项 | 当前值 | 状态 |
|----------------|----------------|------------|-------|--------|
| γ_ENV = 0.992 | Table VI | `algorithm.gamma` | 0.992 | ✅ |
| τ = 0.005 | Table VI | `algorithm.tau` | 0.005 | ✅ |
| Batch size = 256 | Table VI | `algorithm.batch_size` | 128（与论文不一致） | ⚠️ |
| η = 3×10⁻⁴ | Table VI | `algorithm.lr` | 3e-4 | ✅ |
| Replay buffer \|B\| = 2×10⁷ | Table VI | `algorithm.replay_buffer_size` | — | ⚠️ 待核查 |
| γ = 0.7（Q-CBF） | §IV, App. C-5 | — | 未配置 | ❌ |

---

## 需要新建或修改的文件清单

### 需要新建的文件

| 文件 | 用途 | 论文出处 |
|------|---------|----------------|
| `algorithm/hcsf/safety_value.py` | 安全值函数 V_φ(x)：网络定义 + Bellman 损失（公式 21） | §V-A, Eq. 21 |
| `algorithm/hcsf/hcsf_filter.py` | HCSF 运行时滤波器：算法 2 —— 采样 2000 个候选动作，求解 OCP（公式 11） | Def. 2, 算法 2 |
| `algorithm/hcsf/lrsf_filter.py` | LRSF 基线：V(x) = 0 时硬切换到 π^♦（公式 7） | Eq. 7, §III-B |
| `algorithm/hcsf/warmup_policy.py` | 名义 + 超车预热策略（奖励：公式 17–20） | App. C-2 |
| `algorithm/hcsf/training_phases.py` | 预热 + 初始化课程逻辑（阶段概率、终止条件） | App. C-1, Table V |
| `algorithm/hcsf/__init__.py` | 包初始化 | — |
| `train_hcsf.py` | 修改后的 train.py：接入多阶段训练 + HCSF | 算法 1 |
| `evaluate_hcsf.py` | 评估脚本：运行 HCSF/LRSF/None 三种滤波器，记录 IM/jerk/ID 指标（公式 12–14） | §VI-B |

### 需要修改的现有文件

| 文件 | 修改内容 | 论文出处 |
|------|---------------|----------------|
| `algorithm/discor/algorithm/sac.py` | 在 Q-target 计算中加入安全裕度 g：`y = (1−γ_ENV)g + γ_ENV min{g, Q_target}`（公式 21） | Eq. 21 |
| `algorithm/discor/replay_buffer.py` | 在 (s,a,r,s',done) 旁额外存储安全裕度 g(x) | 算法 1 第 9 行 |
| `assetto_corsa_gym/AssettoCorsaEnv/ac_env.py` | (1) 在 g(x) 裕度函数中加入对手有符号距离；(2) 在 `step()` 返回中暴露 g(x)；(3) 添加对手邻近 + 重刹车提前终止开关 | §V-C-3/4, App. C-1 |
| `config.yml` | 新增 HCSF 配置块：`gamma_cbf: 0.7`、预热阶段参数（Table V）、初始化阶段参数 | §IV, App. C, Tables V–VI |
| `train.py` | 增加多阶段训练循环（warmup → initialization → training） | 算法 1, App. C |

### 复现优先级顺序

```
Phase 1 — 基础（让 g(x) 进入训练循环）：
  1. 修改 ac_env.py       → 从 step() 暴露 g(x)
  2. 修改 replay_buffer.py → buffer 中存储 g
  3. 修改 sac.py           → 在 Q-target 中使用 g（公式 21）

Phase 2 — 安全值函数：
  4. 新建 safety_value.py  → V_φ(x) 网络 + 损失

Phase 3 — 运行时滤波器：
  5. 新建 hcsf_filter.py   → OCP 求解器（算法 2）
  6. 新建 lrsf_filter.py   → LRSF 基线

Phase 4 — 训练课程：
  7. 新建 warmup_policy.py  → 奖励公式 17–20
  8. 新建 training_phases.py → 阶段逻辑

Phase 5 — 评估：
  9. 新建 evaluate_hcsf.py  → 指标（公式 12–14）
```
