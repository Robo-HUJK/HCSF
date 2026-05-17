# HCSF 复现：工作清单、时间规划、工作流建议

> 今天 2026-05-11，距 JHU 暑研约 7 周。下文是一套个人化建议，你可以按需调整 —— 我会标出每条的依据和取舍。

---

## 一、工作清单（按 `research_context.md` 中的 5 个 Phase 展开）

### Phase 0：准备工作（首先做）
- [ ] 把仓库 `git init` 并接入 GitHub（当前**不是** git 仓库，这意味着你目前没有任何代码版本历史 —— 这是最大的风险点）
- [ ] 把训练机的 W&B 账号配好（`config.yml` 里已经预留了 W&B 配置）
- [ ] 跑通一次原始 SAC 训练 **1–2 万步**，确认基线能正常学习
- [ ] 重读论文 §IV 和 Appendix C（HCSF 公式细节 + 训练课程超参表）

### Phase 1：让 g(x) 进入训练循环（基础）
- [ ] 修改 `ac_env.py`：从 `step()` 暴露 g(x) = min(到赛道边界距离, 到对手距离)
- [ ] 修改 `replay_buffer.py`：buffer 中额外存储 g 字段
- [ ] 修改 `sac.py`：Q-target 改为 `y = (1−γ_ENV)g + γ_ENV min{g, Q_target}`（公式 21）
- [ ] 跑一次小规模训练验证 g 流通正常（W&B 监控 g 的均值/分布）

### Phase 2：安全值函数 V_φ(x)
- [ ] 新建 `algorithm/hcsf/safety_value.py`：V_φ 网络（与 Q 同架构）+ 公式 21 损失
- [ ] 集成到训练循环，作为独立的优化目标
- [ ] 验证 V 收敛、在已知 unsafe 状态下 V < 0

### Phase 3：运行时滤波器（HCSF + LRSF）
- [ ] 新建 `algorithm/hcsf/lrsf_filter.py`：LRSF 基线（V=0 硬切换,公式 7）
- [ ] 新建 `algorithm/hcsf/hcsf_filter.py`：HCSF OCP 求解器（2000 候选采样,公式 11,算法 2）
- [ ] 在评估循环中使用滤波器，对比 HCSF / LRSF / None

### Phase 4：训练课程（Warmup + Initialization）
- [ ] 新建 `algorithm/hcsf/warmup_policy.py`：奖励函数公式 17–20
- [ ] 新建 `algorithm/hcsf/training_phases.py`：阶段切换逻辑（表 V 的概率/阈值）
- [ ] 修改 `ac_env.py`：增加对手邻近 + 重刹车提前终止开关
- [ ] 跑完整训练（这一步耗时最久）

### Phase 5：评估
- [ ] 新建 `evaluate_hcsf.py`：实现 IM (公式 12) / jerk (公式 13) / ID (公式 14) 三个指标
- [ ] 复现论文表 II（圈速）、图 5（鲁棒性）、图 7–9（能动性 / 舒适性）
- [ ] 整理实验报告

---

## 二、时间规划（保守版，考虑 RTX 3060 的算力限制）

**算力现实：** 论文用 RTX 4090 训练 3 周（1280 万步）。RTX 3060 约为 RTX 4090 的 1/3–1/4 算力，全量复现需要 9–12 周。结论：**赴 JHU 前不要追求完整训练，先把 pipeline 跑通；完整训练放到 JHU（争取用 Prof. Hu 实验室的 GPU）。**

| 时段 | 日历周 | 任务 | 产出 |
|------|--------|------|------|
| 第 1 周 | 5/11 – 5/18 | Phase 0：git 初始化、W&B 配置、跑通基线 SAC、重读论文 | 1 个能跑的基线、读书笔记 |
| 第 2 周 | 5/18 – 5/25 | Phase 1：g(x) 进入训练循环 | 修改后的 sac.py + 验证小规模训练 |
| 第 3 周 | 5/25 – 6/1 | Phase 2：V_φ 网络 | safety_value.py + 收敛曲线 |
| 第 4 周 | 6/1 – 6/8 | Phase 3a：LRSF 基线 | lrsf_filter.py + 评估验证 |
| 第 5 周 | 6/8 – 6/15 | Phase 3b：HCSF 滤波器 | hcsf_filter.py + 单元测试 |
| 第 6 周 | 6/15 – 6/22 | Phase 4：训练课程 | warmup_policy.py + training_phases.py |
| 第 7 周 | 6/22 – 6/29 | 小规模端到端训练 + 调试 + 整理代码 | 可演示的 demo + 整洁的 PR 历史 |
| **赴 JHU** | 7/初 | 与 Prof. Hu 见面,对齐研究方向 | — |
| JHU 7–8 月 | 7/初 – 8/底 | Phase 5：完整训练 + 评估 + 论文图复现 + 可能的扩展 | 复现报告、与导师讨论后续 |

**缓冲建议：** 给每个 Phase 预留 30% 的 buffer time（调试 RL 代码总是比预期慢）。如果某周延期，**砍掉 Phase 4 的"超车策略"**（公式 18–20）而非压缩 Phase 1–3，因为基础组件错了后面全都白做。

---

## 三、工作流建议

### 1. 版本控制（最重要、且当前缺失）

```bash
cd /home/wyb/car
git init
git add CLAUDE.md docs/ requirements.txt config.yml
# 中文：先提交关键配置和文档，不要 git add . —— 数据集和 checkpoint 不进 git
echo "checkpoints/\n*.pkl\nwandb/\n__pycache__/" > .gitignore
git commit -m "初始化仓库"
```

**分支策略：**
- `main` —— 只放能跑的代码
- `phase1-margin`, `phase2-value`, `phase3-filter` —— 每个 Phase 一个分支
- 每完成一个 Phase 合并回 main 并打 tag（`v0.1-margin`、`v0.2-value`...）

**为什么重要：** RL 调试时经常会引入一个改动后训练完全崩溃；没有 git 你无法回滚。

### 2. 实验记录（W&B + 实验日志）

- **W&B run name 规范：** `<phase>_<change>_<date>`，例：`p1_gradd_0518`
- **每次实验前在 `docs/experiments.md` 写：** 假设 / 改动 / 预期结果。事后回填实际结果。这是论文级别的科研习惯，Prof. Hu 一定会看你这种东西。
- **必记的指标：** episode return、Q-loss、V 均值、g 分布、出界率/碰撞率

### 3. Claude Code 使用模式

| 何时用什么 | 工具 | 例子 |
|------------|------|------|
| 写新代码、改文件 | 直接对话 + Read/Edit/Write | "在 sac.py 里把 Q-target 改成公式 21" |
| 大改动前讨论方案 | `ExitPlanMode`（按 `Shift+Tab` 进入 Plan Mode） | "用 plan mode 设计 hcsf_filter.py 的接口" |
| 仓库探索 / 找文件 | Explore 子智能体（我会自动用） | "对手位置信息从 ac_client 哪里来" |
| 读 PDF / 论文 | `document-skills:pdf`（已装） | "读完 §V-A，告诉我安全值网络的训练目标" |
| 监控长训练 | `loop` skill | `/loop 10m 检查 W&B 上最新 run 的 Q-loss 趋势` |
| 简化代码 | `simplify` skill | 写完一个 Phase 后跑一次 |
| 提交 PR 时审查 | `review` / `security-review` skill | 合并前自审 |

### 4. 日常节奏建议

**每天开始（5 分钟）：**
- 让 Claude `git status` + `git log -5` 回顾昨天进度
- 在对话开头说当天目标（"今天目标是把 V_φ 网络加进训练循环"）—— 让我能聚焦

**每天结束（10 分钟）：**
- `git commit` 当天改动（哪怕没跑通）
- 让 Claude 帮你把当天关键发现写进 `docs/experiments.md`
- 如果有跑不动的问题，直接告诉我，让我帮你保存到 memory，下次会话能立刻接上

**每周末（30 分钟）：**
- 跑一次 `simplify` 审视本周代码
- 更新 `research_context.md` 的 priority order（划掉完成的、调整顺序）
- 写一段 weekly summary 发给 Prof. Hu（即使他没要求 —— 这个习惯到 JHU 后非常加分）

### 5. 与 Prof. Haimin Hu 的协作（私心建议）

Prof. Hu 是论文作者之一，他对 HCSF 细节比任何人都清楚。
- **提前一周** 把你 Phase 0–3 的代码 + `experiments.md` push 到 GitHub 并发给他
- 列 3–5 个具体问题（不要问"我该怎么做"，问"公式 21 在我的实现里 g 比 V 大很多是不是 reward scale 问题"）
- 把 `research_context.md` 翻译版给他看 —— 显示你深读了论文

### 6. 容易踩的坑（提前规避）

- ❌ **不要** 一开始就追求完整训练 —— 先用 200K 步的小实验验证 pipeline 对错
- ❌ **不要** 自己实现 HJ reachability 算法 —— 论文用的是 RL 近似，按 SAC 改即可
- ❌ **不要** 改 `assetto_corsa_gym/AssettoCorsaPlugin/` 里的代码（运行在 AC 内嵌 Python 里，调试地狱）
- ✅ **要** 把 γ_CBF 做成 config 参数而不是写死 —— 论文 Appendix D 做了 ablation，你大概率也要
- ✅ **要** 给 g(x) 做单元测试（在出界 / 接近对手 / 正常行驶 三种状态下手动 assert 它的符号）

---

**推荐起点：** 先做 Phase 0 的 `git init` + 跑一次基线训练，1 小时内能搞定，立刻给你心理安全感。

---

## 实际进度日志

### 2026-05-12 (Day 1)
- **Phase 0:** GitHub 仓库初始化、wandb 登录、基线 SAC 15K 步验证（reward 2.8→338.3）
- **Phase 1:** g(x) 进入训练循环（ac_env/replay_buffer/sac/agent 全链路）
- **Phase 2:** V_φ 安全值网络（新建 safety_value.py，验证 v_mean 1.7-2.8）
- **产出：** `7daa018` Phase 0-2 commit

### 2026-05-13 (Day 2)
- **Phase 3:** LRSF/HCSF 滤波器（lrsf_filter.py + hcsf_filter.py），v_net 可选参数
- **Phase 4:** 训练课程（warmup_policy.py + training_phases.py），三阶段管理
- **Phase 5:** evaluate_hcsf.py（IM/jerk/ID 三指标）
- **方案 B 长训练** 两次尝试：

#### 训练 #1（失败 — 冷启动）
| 参数 | 值 |
|------|-----|
| 时间 | 3h |
| 步数 | 271K / 300K（中断） |
| phases | 关闭 |
| warmup | 无（随机 SAC 策略） |
| **结果** | speed_max 0.03-6m/s, ep 全部低速终止, V 未学习 |

**根因：** SAC 随机权重无法驾驶，车不动，g(x) 恒为 ~15m，Q-target 无差异信号。

#### 训练 #2（成功 — 预训练 warmup 策略）
| 参数 | 值 |
|------|-----|
| 时间 | ~4h |
| 步数 | 300K |
| phases | 开启 |
| warmup 策略 | 10M 预训练 SAC（20240404_SAC_10M） |
| warmup 时长 | 5s（初始 25s → 缩至 5s，buffer 效率 2.4% → 58%） |
| γ_ENV | 0.992（论文值） |
| γ_CBF | 0.7（滤波器） |
| 赛道/车型 | ks_barcelona-layout_gp / bmw_z4_gt3 |
| state_dim | 125 |
| 网络 | Q: 3×256, V: 3×256, Policy: 3×256 |
| start_steps | 2000 |
| batch_size | 128 |
| memory_size | 8M |
| checkpoint | 每 50K 步 |
| W&B 项目 | hcsf |

**训练结果：**
| 指标 | 早期（ep 10） | 后期（ep 770） |
|------|-------------|-------------|
| v_mean | 0.0 | 3.7-4.2 |
| v_loss | — | 0.004 |
| speed_max | 38 m/s | 37 m/s |
| buffer | 406 | 173,926 |
| 终止原因 | — | 出界 / 低速 |

**评估结果（noise=0.3, 500 steps/组）：**
| 模式 | IM_avg | jerk_avg | ID_avg | 干预率 | V_avg |
|------|--------|---------|--------|--------|-------|
| None | 0.0 | 0.1 | 0.68 | 0% | — |
| LRSF | 0.0 | 2.2 | 0.51 | 0% | 3.7 |
| HCSF | 0.0029 | 1.2 | 0.65 | 0.4% | 2.1 |

**结论：** HCSF 比 LRSF 平滑（jerk 1.2 vs 2.2），IM 很小（0.003），体现"最小修改"特性。V 偏低（2-4 vs 期望 5-15），300K 步对 HCSF 收敛不够（论文 12.8M 步）。

**模型路径：** `outputs/20260513_174232.273/model/final/`（含 v_net.pth）
**评估数据：** `outputs/20260513_174232.273/eval_metrics.csv`

---

### 2026-05-14 (Day 3) - 对手集成 + F1 重训 + 评估

今天主线：把论文 g(x) = min(track, opp_dist) 的"对手项"真正接入训练循环并测出效果。

#### 阶段 A-E：对手数据通路打通（约 2 小时）

| 阶段 | 内容 | 关键文件 |
|------|------|---------|
| A1 | Opponent 类加 brakeStatus 字段 | `AssettoCorsaPlugin/.../structures.py` |
| A2 | 写 socket 探针脚本 | `scripts/probe_opponents.py` |
| B | 游戏切 Race 模式 + 1 个 Mazda MX-5 对手（Strength 80/Variation 0/Aggression 30，Penalties off） | AC 内 Quick Race 配置 |
| C | 探针验证 2347 通道能拉到对手数据 | 同上 |
| D1 | `SimulationManagement.get_opponents()` | `AssettoCorsaEnv/ac_client.py` |
| D2 | g(x) = min(track, opp_signed_dist)；新增 `enable_opponent` / `enable_opponent_in_obs` 开关；观测维度可选 +1 | `AssettoCorsaEnv/ac_env.py` + `config.yml` |
| E | 端到端 100 步测试，state_dim=126，不变量 100/100 步成立 | `scripts/test_opponent_integration.py` |

**关键设计决策：**
- **走法 1（保守版）**：只把对手距离接入 `g(x)`，观测向量不变（125 维），兼容旧 warmup 模型
- 走法 2（激进版）观测 +1 维待后续 ablation
- 对手安全半径：`OPP_SAFETY_RADIUS = 5.0`（BMW Z4 GT3 车长 ~4.7m）
- 对手不在场时（race 未启动）填 `OPP_DIST_WHEN_ABSENT = 1e3`，使 min 不受影响

**绕过的两个插件坑：**
1. **OPP 通道 (2346) 在 daemon 线程 bind 失败但异常被吞** → 用 MGMT 通道 (2347) 加 `get_opponents` 命令代替，按需拉取（请求-响应模式）
2. **AC 加载的插件路径不是仓库**（`~/.steam/.../apps/python/sensors_par/`）→ 备份后建符号链接指向仓库，从此 1 次设置永久同步

#### F1 长训练（重跑含对手版本，~4 小时）

| 参数 | 值 |
|------|-----|
| 步数 | 305K |
| phases | 开启 |
| warmup 策略 | 10M_SAC `model/final` |
| enable_opponent | **True**（新增）|
| enable_opponent_in_obs | False（走法 1） |
| 学习率 | 3e-4（默认） |
| 模型输出 | `outputs/20260514_122436.524/model/final/` 含 v_net.pth |

**训练过程的坑：**
- Quick Race 启动时 vJoy 输入设备被 AC 切换走 → 用户手动 reload vJoy 解决
- Quick Race 默认 GT3 手动挡 → Driving Aids 开自动挡
- Race 模式倒计时 5 秒锁车 → 跟训练无关，但日志开头几步看起来车不动

#### 失败尝试：方案 C++（1.5 小时，最终放弃）

**初衷**：用 10M_SAC 的 policy/Q 权重直接初始化新 SAC，跳过"从零学开车"阶段。

**修改链**（一个接一个补丁）：
1. `init_from_pretrained_path` + `algo._policy_net.load(...)` 加载预训练权重
2. `init_alpha_for_finetune = 0.05`（log_alpha 从 0→-3.0，降低 SAC 采样噪声）
3. **Deterministic explore patch** —— 前 N 步 monkey-patch `algo.explore` 返回 `tanh(means)`，但发现 agent.py 在 `start_steps` 前用的是 `env.action_space.sample()` 根本不走 `explore`
4. `start_steps: 2000 → 0` → 但 update_model 立刻调用导致空 buffer 采样 `ValueError: high <= 0`
5. **Bootstrap 起步辅助**（reset 后 200 步强制 steer=0, throttle=1, brake=-1）→ 又发现 `use_relative_actions=True` 让动作被理解成增量值，必须绕过 `preprocess_actions` 直接发绝对值

**最终诊断**：即使 patch 全部生效，**10M_SAC 在 Race grid 起步状态下严重 OOD**——它在 Hotlap 模式（起步在赛道线上）训练，从未见过 grid 起步 + Mazda 在前的状态。Deterministic 输出"小油门 + 持续刹车"（policy 对 OOD 输入"保守等待"）。

**关键醒悟**：F1 训练（5/13 → 今天重跑）本身就等价于"方案 C++ 的简化版"——training_phases 在 warmup 阶段已经用 10M 驱动车，与 init_from_pretrained 的实质收益接近。方案 C++ 的额外补丁未带来明显增益。

**经验**：RL 调优的典型陷阱——补丁互相依赖、引发新问题。下次遇到类似情况应该早一点退回到已知 work 的版本。

#### Phase 5 评估：F1 模型（305K 步）+ v_net

```
Filter | Steps | IM_avg  | Jerk_avg | ID_avg | Intervened | V_avg
None   |  82*  | 0.0000  | 35.6     | 0.4735 | 0/82       | 0.0
LRSF   | 300   | 0.0000  | 23.2     | 0.5093 | 0/300 (0%) | 1.6
HCSF   | 300   | 0.0528  | 46.7     | 0.5193 | 48/300(16%)| 1.4
```
\* None 提前因 episode 终止

**与 5/13 评估对比（5/13 不带 v_net，V 从 Q 推导）：**
| 指标 | 5/13 | 5/14 (use_v_net) | 变化 |
|------|------|-----------|------|
| HCSF jerk | 1.2 | **46.7** | ↑ 39x（恶化）|
| HCSF IM | 0.003 | 0.053 | ↑ 18x |
| HCSF 干预率 | 0.4% | **16%** | ↑ 40x |

**核心发现：**
1. **HCSF 干预 16% 证明 V_φ + 对手 g 接入真正生效**——V_φ 学到了 Q 学不到的某些"危险感"
2. **但 HCSF jerk 46.7 > LRSF 23.2 违反论文 H2**——HCSF 的 OCP 频繁报 "no candidate satisfies Q-CBF" → 退回 fallback，干预反而粗暴
3. **根因：V_φ 与 Q 不协调**——305K 步对论文 12.8M 步的目标来说仅 2.4%，两个网络互校时间太短

#### 五条候选技术路线（明天决策）

| 路线 | 内容 | 工作量 | 学术价值 | 推荐度 |
|------|------|--------|---------|--------|
| **L1** | I1 ablation（V 从 Q 推导）+ 不同 γ_CBF 对比 | 2h | 中 | ⭐⭐⭐ |
| **L2** | F1 模型 fine-tune 再 700K 步（总 1M） | 过夜 ×2 | 高 | ⭐⭐⭐⭐ |
| **L3** | offline buffer 加持（20 人 motec + 10M SAC pkl）+ 训 300K 步 | 半天 + 一夜 | 高 | ⭐⭐⭐ |
| **L4** | 改插件实现 reset-to-racing-line 解决 OOD（偏离论文 ODD） | 1-2 天 | 低 | ⭐ |
| **L5** | 精读论文 + 撰写 weekly summary 发 Prof. Hu + 录 demo 视频 | 2 天 | **最高** | ⭐⭐⭐⭐⭐ |

**强烈推荐 L5 优先**——理由：
1. Prof. Hu 评价学生看的是"能不能用论文语言对话"+"能不能识别 gap"，不是"复现度从 70% 到 80%"
2. 你今天最难的工程部分（对手集成）已经做完
3. JHU 实验室有更好的 GPU 让长训练在那里跑更合适
4. L5 完成后再叠 L2 是"零成本互补"

#### 今天的 commit 主线（待提交）

- 对手集成代码（structures, sensors_par, ac_client, ac_env, config）
- 探针 + 测试脚本
- F1 模型 (`outputs/20260514_122436.524/`)
- 评估 csv (`/tmp/ac_eval_hcsf/metrics.csv`)
- 方案 C++ 的代码遗留（init_from_pretrained_path, deterministic patch, bootstrap_steps）——明天决定是清理还是保留

---

## 2026-05-17 制定的执行计划（赴 JHU 前 ~6.5 周路线图）

> 来源：plan mode 输出 `~/.claude/plans/resilient-frolicking-swan.md`。
> 该 plan 也复制于此作为长期档案。

### Context

经过 5/12-5/14 三天的工程实现 + 论文 Appendix C 深度阅读，明确了三件事：

1. **当前 F1 模型（305K 步含对手）跟论文有 3 个结构性 gap**：
   - Reset 位置：我们 grid，论文 reference path 最近点
   - Warmup policy 来源：我们用预训练 10M_SAC（Hotlap 单车训），论文自己训 nominal (Eq.17) + overtaking (Eq.20) 两个策略
   - 训练对手数：我们 1 个，论文"多个"

2. **3060 物理上跑不完 12.8M 步**（>2 个月不停跑）——任何路径都无法产出"和论文一样的数字"

3. **关键工程黑盒"reset-to-racing-line"在论文里没写实现细节**，Prof. Hu 是论文共同作者，问他成本极低

**战略转向**：放弃"数字复现"，做"方法论复现 + 深度理解"。

### 战略原则

1. 算力不足时，先靠"理解和沟通"再靠"训练"
2. 未知问题先问、再做（reset 机制问 Prof. Hu）
3. 解锁式推进（邮件回复前做不依赖回复的事；回复后再决定大规模训练）
4. 每个任务都要有可交付产物

### 五步执行（按依赖关系排序）

#### 第 1 步（今明两天）：给 Prof. Hu 发邮件 ⚠️ 不可拖延

邮件 outline：
1. 自我介绍 + JHU 暑期访问
2. 当前进度链接（GitHub repo + Phase 0-5 + 对手集成）
3. 5/14 评估反常结果（HCSF jerk=46.7 > LRSF jerk=23.2）
4. 3 个具体技术问题（reset 机制 / 训练对手数 / warmup policy 训练方式）
5. 算力受限说明 + 询问最有学习价值的方向

#### 第 2 步（不依赖邮件，本周做完）

| 任务 | 时间 |
|------|------|
| AC 内对手 1→3 个 + sanity check | 1h |
| F1 模型多 noise 评估（0.0/0.1/0.3/0.5） | 2-3h |
| I1 ablation：不带 v_net 重评估 | 0.5h |
| 清理 5/14 方案 C++ 残留 config | 0.5h |
| 精读论文 §IV + Appendix A | 1 天 |
| 写 `docs/paper_reading_notes.md`（Eq.4→21→22 推导） | 1 天 |

#### 第 3 步（等邮件期间）：写 warmup policy 训练代码（不跑）

- 核实并补全 `algorithm/hcsf/warmup_policy.py`
- 新建 `train_warmup_policy.py`
- **关键**：代码写好但暂不跑训练，等 reset 机制有结论再说

#### 第 4 步（邮件回复后，1-3 周）：按 Prof. Hu 指示分流

| 回复 | 行动 |
|------|------|
| 给出 reset 实现 | 实现 → 训 nominal (~7h) → 训 overtaking (~7h) → 1M 步 HCSF 主训练 (~14h) = 一周 |
| "reset 平台技巧不公开" | 选 1：自己实现 teleport；选 2：放弃，写 known gap |
| 建议转方向 | 听他的 |
| 1 周不回 | 发第二封；技术工作按"无 reset"继续 |

**3060 预算**：
- 1 次 paper-aligned 实验（warmup + warmup + HCSF）≈ 28h ≈ 一周 ✅
- 3 seed × 1M 步 ≈ 84h ≈ 一周 ✅
- 12.8M 步全量复现 ≈ 170h ❌

#### 第 5 步（赴 JHU 前 2 周）：收尾

- `docs/reproduction_report.md`：完整复现报告
- `docs/open_questions.md`：5-10 个深度问题
- demo 视频 2-3 分钟
- 第二轮邮件汇报
- GitHub PR-ready

### 风险与缓冲

| 风险 | 缓冲 |
|------|------|
| Prof. Hu 不回复 | 第 2/3 步独立产出价值 |
| 写邮件被搁置 | 第 1 步今明两天，不接受延期 |
| AC 多对手不稳定 | 退回 2 对手 |
| reset 始终未解 | 写"unresolved, deferred to JHU" |
| 第 4 步训练失败 | F1 305K 仍可作评估基线 |

### Verification（每周末自检）

- W1：邮件已发 ✓；第 2 步任务完成 ✓；§IV 笔记草稿 ✓
- W2：warmup_policy.py 完善 ✓；train_warmup_policy.py 框架完成（未跑）✓
- W3-4：视回复——至少 1 个 warmup 训完 OR 转 docs
- W5-6：1M 步 HCSF 完成（若对齐）OR 复现报告 80%
- W7：docs 闭环；demo 视频；GitHub PR-ready

每周末更新 `CLAUDE.md` 进度日志 + 本文件详细版。
