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
