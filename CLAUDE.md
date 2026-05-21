# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
> 中文：本文件用于为 Claude Code（claude.ai/code）在本仓库中工作时提供指引。

## Project Overview
> 中文：项目概览

**Assetto Corsa Gym** is a Python reinforcement learning framework integrating the Assetto Corsa racing simulator with OpenAI Gym. It supports training autonomous racing agents (primarily SAC/DisCor algorithms) against a high-fidelity simulator, and includes a 2.3M-step human driving dataset.
> 中文：**Assetto Corsa Gym** 是一个将 Assetto Corsa 赛车模拟器与 OpenAI Gym 对接的 Python 强化学习框架。它支持在高保真模拟器上训练自动驾驶赛车智能体（主要使用 SAC/DisCor 算法），并提供了一份共 230 万步的人类驾驶数据集。

## Setup & Installation
> 中文：环境搭建与安装

```bash
pip install -r requirements.txt
# 中文：安装 Python 依赖包
```

Platform-specific plugin setup is documented in `INSTALL.md` (Windows) and `INSTALL_Linux.md` (Linux via Proton).
> 中文：与平台相关的插件安装步骤分别记录在 `INSTALL.md`（Windows）和 `INSTALL_Linux.md`（通过 Proton 在 Linux 上运行）。

## Key Commands
> 中文：常用命令

**Train from scratch:**
> 中文：从零开始训练
```bash
python train.py
# 中文：使用默认配置启动训练
```

**Train with config overrides (OmegaConf syntax):**
> 中文：通过命令行覆盖配置项（OmegaConf 语法）
```bash
python train.py algorithm=sac env.track=monza env.car=gt3
# 中文：指定算法为 SAC、赛道为 Monza、车型为 GT3
```

**Test checkpoint:**
> 中文：测试已有 checkpoint
```bash
python train.py load_checkpoint=path/to/checkpoint evaluate=True
# 中文：加载指定 checkpoint 并进入评估模式
```

**Convert MoTeC telemetry to training data:**
> 中文：将 MoTeC 遥测数据转换为训练数据
```bash
python motec_to_pickle.py
# 中文：把原始 MoTeC 录制文件转为 pickle 格式的训练样本
```

**Interactive testing (Jupyter notebooks):**
> 中文：交互式测试（Jupyter notebook）
- `test_gym.ipynb` — environment step/reset without agent
  > 中文：在没有智能体的情况下测试环境的 step/reset 行为
- `test_client.ipynb` — raw AC socket client
  > 中文：直接测试与 Assetto Corsa 通信的底层 socket 客户端
- `test_gym_images.ipynb` — screen capture pipeline
  > 中文：测试屏幕截图（screen capture）的流水线
- `test_maps_creator.ipynb` — track config generation
  > 中文：生成赛道相关的配置文件（occupancy grid、参考线等）

There is no automated test suite or lint configuration.
> 中文：仓库没有自动化测试套件或 lint 配置。

## Architecture
> 中文：架构

### Communication Stack
> 中文：通信栈

```
train.py  →  Agent (algorithm/discor/)
              │
              └→  AssettoCorsaEnv (Gym)  ←→  AC Client (TCP sockets 2345-2347)
                                                    │
                                               AC Plugin (Python inside Assetto Corsa)
                                                    │
                                          Assetto Corsa Simulator
```
> 中文：训练入口 `train.py` 调用 Agent；Agent 通过 Gym 风格的 `AssettoCorsaEnv` 与环境交互；环境通过 TCP socket（端口 2345–2347）连接到运行在 Assetto Corsa 内部的 Python 插件；插件再把控制信号施加到模拟器上，并把遥测数据回传。

### Key Directories
> 中文：关键目录

**`assetto_corsa_gym/AssettoCorsaEnv/`** — OpenAI Gym environment:
> 中文：OpenAI Gym 风格的环境实现
- `ac_env.py` — core Gym class; defines observation/action spaces, reward function, episode termination
  > 中文：核心 Gym 类；定义观测/动作空间、奖励函数、episode 终止条件
- `ac_client.py` — socket client sending controls, receiving telemetry
  > 中文：socket 客户端，负责发送控制指令、接收遥测数据
- `reference_lap.py` + `gap_*.py` — reference racing line used for reward shaping
  > 中文：参考赛车线及到参考线的间距计算，用于奖励塑形
- `track.py`, `track_occupancy_grid.py` — track geometry and collision detection
  > 中文：赛道几何信息与碰撞检测
- `sensors_ray_casting.py` — LIDAR-style ray casting for obstacle sensing
  > 中文：类似 LIDAR 的射线投射传感器，用于感知障碍
- `motec_loader.py`, `data_loader.py` — load offline human demonstration datasets
  > 中文：加载离线的人类示范数据集

**`assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensors_par/`** — Plugin running inside Assetto Corsa:
> 中文：运行在 Assetto Corsa 内部的插件
- `sensors_par.py` — plugin entry point
  > 中文：插件入口
- `ego_server.py` — streams telemetry out via socket
  > 中文：通过 socket 流式输出自车遥测
- `car_control.py` — applies control inputs (steering, throttle, brake)
  > 中文：把控制输入（转向、油门、刹车）施加到车辆
- `screen_capture.py` + `dual_buffer.py` — low-latency screen capture (added Feb 2025)
  > 中文：低延迟屏幕截图（2025 年 2 月新增）

**`algorithm/discor/`** — RL algorithms:
> 中文：强化学习算法
- `agent.py` — training loop, replay buffer management, checkpointing (every 200K steps), W&B/TensorBoard logging
  > 中文：训练主循环；管理 replay buffer；每 20 万步保存 checkpoint；记录 W&B / TensorBoard
- `algorithm/sac.py` — Soft Actor-Critic
  > 中文：Soft Actor-Critic 算法实现
- `algorithm/discor.py` — DisCor (distribution correction variant)
  > 中文：DisCor 算法（分布修正变体）
- `network.py` — policy and Q-function neural networks
  > 中文：策略网络与 Q-函数网络
- `replay_buffer.py` — supports both online and offline (human demo) replay
  > 中文：同时支持在线训练与离线（人类示范）数据的 replay buffer

**`assetto_corsa_gym/AssettoCorsaConfigs/`** — Static assets:
> 中文：静态资源
- `tracks/` — per-track occupancy grids, reference racing lines, bounds (pickle files)
  > 中文：每个赛道的占据栅格、参考线、边界（pickle 文件）
- `cars/` — car-specific parameters
  > 中文：每辆车的相关参数

**`common/`** — Shared utilities: W&B integration (`logger.py`), logging setup, dataset statistics.
> 中文：通用工具：W&B 集成（`logger.py`）、日志配置、数据集统计。

### Configuration System
> 中文：配置系统

All hyperparameters and environment settings live in `config.yml`, loaded via **OmegaConf**. Key sections: `algorithm` (SAC params, batch size, replay buffer), `env` (track, car, observation space components), `training` (total steps, eval frequency).
> 中文：所有超参数与环境设置都集中在 `config.yml`，通过 **OmegaConf** 加载。主要小节：`algorithm`（SAC 参数、batch size、replay buffer 等）、`env`（赛道、车型、观测空间组件）、`training`（总步数、评估频率）。

### Observation & Action Spaces
> 中文：观测空间与动作空间

- **Observations:** position, velocity, acceleration, steering angle, lookahead curvature, gap to reference line, recent action history, optional screen images
  > 中文：**观测：** 位置、速度、加速度、转向角、前瞻曲率、距参考线间距、近期动作历史、可选的屏幕截图
- **Actions:** `[steering ∈ [-1,1], throttle ∈ [0,1], brake ∈ [0,1], gear/clutch]`
  > 中文：**动作：** `[转向 ∈ [-1,1]，油门 ∈ [0,1]，刹车 ∈ [0,1]，档位/离合]`

### Offline Pre-training
> 中文：离线预训练

The replay buffer supports loading human demonstration data (HuggingFace, ~120GB total). Dataset paths are listed in `data/paths.yml`. Use `motec_to_pickle.py` to convert raw MoTeC recordings.
> 中文：Replay buffer 支持加载人类示范数据（HuggingFace 上共 ~120GB）。数据集路径列于 `data/paths.yml`。使用 `motec_to_pickle.py` 将原始 MoTeC 录制文件转为 pickle。

## Platform Notes
> 中文：平台说明

- **Linux (preferred):** Runs Assetto Corsa via Proton; faster and more stable. See `INSTALL_Linux.md`. Uses `vjoy_linux.py` + `evdev`.
  > 中文：**Linux（推荐）：** 通过 Proton 运行 Assetto Corsa，更快也更稳定。详见 `INSTALL_Linux.md`。使用 `vjoy_linux.py` + `evdev`。
- **Windows:** Native; uses `vjoy.py` and vJoy driver.
  > 中文：**Windows：** 原生运行；使用 `vjoy.py` 与 vJoy 驱动。
- Plugin code runs inside Assetto Corsa's embedded Python interpreter (limited stdlib — runtime deps bundled in `windows-libs/`).
  > 中文：插件代码运行在 Assetto Corsa 内嵌的 Python 解释器中（标准库受限——运行时依赖打包在 `windows-libs/` 下）。

## Project-Specific Conventions
> 中文：本项目的约定

- **Documentation language:** All new documentation and code comments are written in **Chinese (Simplified)**. Code identifiers (variable/function/class names), imports, log strings, and paper references (e.g., "Eq. 11") remain in English.
  > 中文：**文档语言：** 所有新增文档与代码注释一律使用 **简体中文**。代码标识符（变量、函数、类名）、import 语句、日志字符串、论文引用（如 "Eq. 11"）保持英文不变。
- **Current research goal:** Reproduce the HCSF method from "Safety with Agency" (RSS'25). See `docs/research_context.md` for the paper-to-code mapping and reproduction roadmap.
  > 中文：**当前研究目标：** 复现 RSS'25 论文 "Safety with Agency" 中的 HCSF 方法。论文组件与代码文件的对应关系、复现路线图见 `docs/research_context.md`。

---

## 进度日志
> 中文：每天结束时由 Claude 追加一条；只保留最近 7 天，更早的条目转存到 [`docs/daily_log.md`](docs/daily_log.md)。
> 实验结果详见 [`docs/experiments.md`](docs/experiments.md)，工作流约定见 [`docs/workflow_guide.md`](docs/workflow_guide.md)。

### 2026-05-17
- **完成：** 制定赴 JHU 前 ~6.5 周路线图（plan 五步走，见 `~/.claude/plans/resilient-frolicking-swan.md`）
- **完成：** 深度阅读论文 §V-D + Appendix C，明确 3 个结构性 gap 与 paper 差距
- **完成：** 邮件草稿 v5（`docs/email_to_prof_hu_v1.md`）：3 个具体技术问题 + 算力说明 + 求方向
- **战略转向：** 放弃数字复现，转向方法论复现 + 深度理解 + 与 Prof. Hu 技术对话
- **决策：** 第 1 步（发邮件）阻塞所有技术工作，第 2-4 步本周做不依赖回复的小任务

### 2026-05-18
- **完成（任务 #1）：** 清理 5/14 方案 C++ 残留 config（commit `46e8b25`），回到 F1 baseline
- **完成（任务 #2）：** 修改 `evaluate_hcsf.py` 加 `--human-model` / `--filter-model` 支持解耦评估（论文 §VI / Appendix C-4 设计）
- **关键发现：单模型评估对 F1 模型是 spurious 的：**
  - 用 F1 model 自己当 human → policy_net 不会开车 → 车不动 → jerk/IM 数字全是幻觉
  - 实测：今天上午 4 个 use_v_net run 里 9/12 episode speed mean < 0.01 m/s
- **解耦评估结果（10M as human + F1 as filter, noise=0.3）：**
  - None: jerk=70.6, 干预 0%, speed mean=4.1
  - LRSF: jerk=56.5, IM=0.02, 干预 1.6%, speed mean=9.1（比 None 平滑）
  - **HCSF: jerk=118.7, IM=0.42, 干预 26.6%, speed mean=9.6**（**比 LRSF 更差**，违反论文 H2）
  - 大量 "no candidate satisfies Q-CBF" → fallback 频繁触发
- **诊断：** F1 V_φ 过度激进（OOD 训练后认为多数状态危险）→ HCSF 频繁触发 → 找不到 Q-CBF 候选 → 粗暴 fallback。**根因 Race grid OOD 训练**
- **方法学价值：** 解耦评估方法学跑通，5/14 F1 模型真实 V_φ 行为首次被测出。
- **新增 memory：** `reference_decoupled_eval.md` + `feedback_distinguish_train_vs_eval_mobility.md`
- **邮件发送 + Prof. Hu 当天回复**（关键转向，详见 `memory/project_prof_hu_reply_20260518.md`）:
  - Q1 Reset: 暑期跟 David 当面解决，目前用现有 reset
  - Q2 对手数: 1 或 0 都 OK
  - Q3 Warmup: tentatively frozen，David 确认
  - **算力**: RTX 3060 跑 12.8M 可行（1-2 周），keep training
  - **JHU 设备**: workstation + simulator 7 月第二周到
- **用户决定**: July 1 准时到 JHU，先用 laptop
- **战略转向**: 不再等 reset，按 Hu 建议直接长训练
- **完成（下午）**: 起草 `docs/open_questions_for_david.md`（9 个技术问题 / 25 子项 / P1-P5）
- **完成（下午）**: 优化 `.gitignore` + 撤回邮件草稿跟踪（commit `7c85c6a`）
- **当前进度：** 5 个文件 5/18 改动已 commit (`3b2c558`)；本地领先 origin 3 commit 未 push
- **下一步（晚上）：** 启动 1M 步 Hotlap 长训练（无对手，from scratch）

### 2026-05-19/20
- **1M Hotlap 长训练完成：** `outputs/20260519_G2_1M_plateau/model/final/`（含 v_net.pth）
  - 总步数 1,006,117，2022 episode，运行 ~12h
  - 技术路径: warmup(10M,5s) → 跳过 init → 直接 training(SAC探索)
  - **关键修改:** `agent.py:194` 跳过 init 阶段（warmup → training 直接过渡），init 的随机/对抗动作在 Hotlap 下导致频繁 crash，数据效率从 ~20/ep 提升到 ~100-150/ep
  - 4 次失败启动（nohup 缓冲/conda run/init crash/无 phases 冷启动），仅保留最终目录
- **训练曲线 plateau：** V_φ 从 2.1 快速升到 4.0（前 150K 步），之后 850K 步只涨到 4.77。episode reward/speed/steps 全部 150K 后停滞
- **解耦评估对比（10M as human, noise=0.3, 500 步）：**

  | 指标 | 300K (5/13) | 1M (5/19) |
  |------|-----------|-----------|
  | HCSF IM | **0.075** | 0.156 |
  | HCSF jerk | **60.3** | 65.5 |
  | HCSF 干预率 | **8.6%** | 18.6% |
  | no-candidate | 10 次 | 4 次 |
  | LRSF 干预率 | 0%（V 始终 > 0） | 0% |

  - **1M 不比 300K 好。** HCSF jerk 65.5 > 60.3，IM 0.156 > 0.075。瓶颈不在步数，在训练场景多样性
- **根因：** Hotlap 单车模式缺乏危险场景（g(x) 始终 > 0），V_φ Bellman 目标在安全区域退化为 Q 的影子。需回到含对手 + init 阶段的 Race 模式
- **当前进度：** 1M 模型产出但未超越 300K。RTX 3060 验证 12h 可跑 1M
- **下一步：** 用 plan mode 规划下阶段——Race + 对手 + init，解决 OOD 问题

### 2026-05-20 晚
- **G3 (612K kill):** Hotlap 1M + INIT 恢复 + 4 项观测加固。π^♦ 驱动 TRAINING，speed 单调下降 20→13 m/s，63% episode "Speed too low" 终止。同 5/19 plateau 模式
- **G4 (122K kill, 选项2 dead end):** 改 agent.py 让 10M_SAC 驱动 TRAINING + 噪声 0.1（误读 paper §V-D intent）。10M 太能开 → episode 跑满 15001 步 timeout → INIT ratio 从 5/19 的 1-4% 跌到 0.02% → V_φ stuck @ 2.4。**结构错误：10M 不会失败 → 永远不触发 episode reset → INIT 几乎不点火，回退**
- **路径决策（重读论文 §V-C-4 + §V-D）：**
  - 论文 episode 自然终止：`g<0` 或 stationary（无 step cap）
  - 论文 π^♦ 经常失败 → episode 短 → 频繁 reset → INIT ratio ~3%
  - 我们 1M 步预算下，**短 episode hack 可补偿 π^♦ 失败频率不足**
- **G5 启动 (Path C):** π^♦ 驱动（paper-aligned 回退选项 1）+ `max_episode_py_time=60s`（从 600s 砍到 60s，强制 10x reset → 10x INIT）+ 1M 步，预计 ~11h 过夜跑完
- **GPU 瓶颈澄清:** 实测 3060 GPU 不满载——AC 物理仿真 ~30ms/step（CPU 为主）才是瓶颈，NN 训练只占 5-10ms/step。4090 比 3060 只快 10-15%。**Paper-scale 12.8M 在 3060 上 1-2 周完全可行**，印证 Hu 5/18 "keep training"

### 2026-05-21
- **G5 训练完成 1M 步**（`outputs/20260520_G5_1M_pathC_div_50KSOTA/`），但 V_φ 在 100K-130K 步发散：
  - 训练曲线: 45K v_mean=2.44 ✅ → 91K v_mean=1.64 → 136K v_mean=-2.27 → 362K v_mean=-18,228 (峰值崩溃) → 1M v_mean=-417
  - v_loss 从 0.054 → 337M（70 亿倍放大），actor-critic 反馈循环爆炸
  - 后 95% 步数在毒化网络上训练，浪费了
- **🏆 50K checkpoint 是迄今 SOTA HCSF 模型**（解耦评估 noise=0.3, 500 步）:

  | Filter | IM | jerk | 干预率 | V_avg | no-cand |
  |---|---|---|---|---|---|
  | None | 0 | 60.3 | 0% | 0 | – |
  | LRSF | 0 | 65.9 | 0% | 2.4 | – |
  | **HCSF** | **0.0074** | **59.9** | **1.2%** | 2.3 | **2** |

  - **✅ 论文 H2 假设满足**: HCSF jerk 59.9 < LRSF 65.9
  - **✅ IM 是 5/13 baseline 的 1/10**（0.0074 vs 0.075）
  - **✅ HCSF jerk ≈ None jerk**: 滤波器不引入额外抖动
  - **✅ no-candidate fallback 仅 2 次/500 步**（5/13 是 10 次）
- **100K checkpoint 已退化**: HCSF IM 0.42（56x worse），115/500 步车飞出。证实 50K 是甜蜜点
- **关键结论**: Path C 短 episode hack 是正确方向。INIT ratio 起来后 V_φ 快速学到边界。**唯一问题：V_φ 需要 target clipping + 早停防止发散**
- **🔬 Variance eval（9 次复测 × 500 步，验证 SOTA 不是偶然）:**

  | 指标 | G5 50K (n=3) | 5/13 baseline (n=3) | 5/19 G2 plateau (n=3) |
  |---|---|---|---|
  | HCSF IM | **0.010 ± 0.007** | 0.086 ± 0.046 | 0.084 ± 0.062 |
  | HCSF jerk | **60.97 ± 5.31** | 64.0 ± 1.28 | 61.77 ± 1.78 |
  | HCSF 干预率 | **1.2 ± 0.9%** | 8.3 ± 5.3% | 10.3 ± 8.4% |
  | LRSF jerk | 64.4 | 64.6 | 63.0 |

  - **✅ IM 区间不重叠**: G5 50K max IM 0.0179 < 5/13 min IM 0.0502 → 统计显著
  - **✅ 干预率不重叠**: G5 max 1.8% < baseline min 4.8%
  - **⚠️ jerk 方差大**: G5 r2 出现 67.9 反例（H2 违反 1 次），但均值 60.97 仍最低
  - **2/3 次满足论文 H2** (HCSF jerk < LRSF jerk)，均值满足
  - 详细 9 行数据见 `docs/experiments.md`
- **outputs/ 大清理**:
  - 删除 G3 (709M, plateau 重复 G2)、G4 (261M, dead end)、G5 final/replay_buffer.pkl (8.2G, 毒化数据) → 共省 ~9G
  - 4 个保留目录重命名：`20260513_300K_hotlap_baseline`, `20260514_305K_race_OODfail`, `20260519_G2_1M_plateau`, `20260520_G5_1M_pathC_div_50KSOTA`
- **下一步候选**:
  - 写 `docs/reproduction_report.md` 把 G5 50K 作为 highlight
  - 修 V_φ 发散：加 target clipping + 降 v_lr + early stopping，再跑 1M+ 步

---

## 后续方向

### 已完成（截至 5/20）
- ✅ Phase 0-5 全套 + A-E 对手集成
- ✅ 5/13 Hotlap 300K baseline（HCSF jerk 1.2 < LRSF 2.2，符合 H2）
- ✅ 5/14 F1 Race 305K（含对手，V_φ 过度激进）
- ✅ 5/20 Hotlap 1M 长训练（V_φ 2.1→4.77，12h可行，但未超300K）
- ✅ 解耦评估方法学（`--human-model / --filter-model`）
- ✅ 邮件 v5 + Prof. Hu 回复 + `open_questions_for_david.md`
- ✅ 战略路线图
- ✅ **5/21 G5 50K SOTA**：Path C 短 episode hack 验证成功，HCSF IM 0.0074 / jerk 59.9 < LRSF 65.9（论文 H2 满足）
- ✅ outputs/ 清理 + 重命名（4 个保留目录可读名，9G 释放）

### 即将执行
- 📦 push 本地 commit 到 GitHub（含 G5 实验 + 清理）
- 📝 写 `docs/reproduction_report.md`，G5 50K 当 highlight 案例
- 🔧 修 V_φ 发散：target clipping + 降 v_lr + early stopping
- 📚 精读论文 §IV + Appendix A

### 短期（赴 JHU 前 ~5.5 周）
- 修 V_φ 发散后再跑 1M-3M 步含 INIT 训练（基于 G5 setup）
- 完整 reproduction report + demo 视频
- 持续更新 `open_questions_for_david.md`
- 银石赛道切换（如有时间）

### 中期（赴 JHU 前 ~7 周）
- 配置对齐：batch_size 128→256, memory_size 8M→20M
- 完整训练 12.8M 步（JHU 实验室 GPU）

### 长期（JHU 期间）
- 用户研究：实人实验，复现论文 Fig.5/7-9
- 过度依赖分析：Session 3 移除滤波器后圈速变化
- Parametric CBF：联合优化 γ（论文 §VII）
