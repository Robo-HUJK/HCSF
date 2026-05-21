# HCSF 复现：工作流指南

> 项目工作流约定、Claude Code 使用模式、与 Prof. Hu 协作 tips。
> 实际进度见 `docs/daily_log.md`，实验结果见 `docs/experiments.md`，论文方法见 `docs/research_context.md`。

---

## 一、Phase 0-5 完成状态

| Phase | 内容 | 状态 |
|---|---|---|
| **Phase 0** | Git 初始化 + W&B 配置 + 基线 SAC 验证 | ✅ 5/12 完成 |
| **Phase 1** | g(x) 进入训练循环（`ac_env` / `replay_buffer` / `sac` / `agent` 全链路）| ✅ 5/12 完成 |
| **Phase 2** | 安全值函数 V_φ(x)（`algorithm/hcsf/safety_value.py`，公式 21）| ✅ 5/12 完成 |
| **Phase 3** | 运行时滤波器（`lrsf_filter.py` + `hcsf_filter.py`）| ✅ 5/13 完成 |
| **Phase 4** | 训练课程（warmup + initialization, Appendix C-1 / Table V）| ✅ 5/13 完成 |
| **Phase 5** | 评估脚本 `evaluate_hcsf.py`（IM / Jerk / ID 三指标）| ✅ 5/13 完成 |
| **A-E 阶段** | 对手集成（g(x) 加入对手距离项）| ✅ 5/14 完成 |
| **方法学** | 解耦评估（`--human-model` / `--filter-model`）| ✅ 5/18 完成 |

所有 Phase 端到端可运行，限制只在算力 / 训练步数。

---

## 二、剩余工作时间规划

**算力现实**：论文用 RTX 4090 训练 3 周（12.8M 步）。实测 3060 GPU 不满载——AC 物理仿真才是瓶颈，4090 比 3060 只快 10-15%。**Paper-scale 12.8M 在 3060 上 1-2 周可行**（Prof. Hu 5/18 邮件确认）。

| 时段 | 任务 | 产出 |
|---|---|---|
| 5/22 - 6/1 | 修 V_φ 发散（target clipping + early stopping）+ 重训 1M-3M 步 | 稳定的 1M+ HCSF 模型 |
| 6/2 - 6/15 | 写 reproduction report + demo 视频 | `docs/reproduction_report.md` |
| 6/15 - 6/30 | 银石赛道切换（如有时间）+ 配置对齐论文（batch_size 256, memory 20M）| 多赛道结果 |
| **JHU 7-8 月** | 完整训练 12.8M 步 + 真人用户研究复现论文 Fig 5/7-9 | 完整复现结果 |

---

## 三、Claude Code 使用模式

| 何时用什么 | 工具 | 例子 |
|---|---|---|
| 写新代码、改文件 | 直接对话 + Read/Edit/Write | "在 sac.py 里把 Q-target 改成公式 21" |
| 大改动前讨论方案 | Plan Mode (Shift+Tab) | "用 plan mode 设计 hcsf_filter.py 的接口" |
| 仓库探索 / 找文件 | Explore 子智能体 | "对手位置信息从 ac_client 哪里来" |
| 读 PDF / 论文 | `document-skills:pdf` | "读完 §V-A，告诉我安全值网络的训练目标" |
| 监控长训练 | `loop` skill | `/loop 10m 检查 W&B 上最新 run 的 Q-loss 趋势` |
| 简化代码 | `simplify` skill | 写完一个 Phase 后跑一次 |
| 提交 PR 时审查 | `review` / `security-review` skill | 合并前自审 |

---

## 四、日常节奏

**每天开始（5 分钟）**:
- `git status` + `git log -5` 回顾昨天进度
- 对话开头说当天目标（"今天目标是…"）→ Claude 能聚焦

**每天结束（10 分钟）**:
- `git commit` 当天改动（哪怕没跑通）
- 让 Claude 把当天关键发现写进 `docs/daily_log.md`
- CLAUDE.md `## 进度日志` 自动维护最近 7 天
- 跑不动的问题告诉 Claude，让它保存到 memory，下次会话能接上

**每周末（30 分钟）**:
- 跑一次 `simplify` skill 审视本周代码
- 更新 `docs/experiments.md` 实验结果
- 写一段 weekly summary 发给 Prof. Hu（即便他没要求 —— 这是 JHU 加分项）

---

## 五、与 Prof. Haimin Hu 协作 tips

Prof. Hu 是论文作者之一，对 HCSF 细节比任何人都清楚。

- **提前一周** 把 Phase 0-3 的代码 + `docs/experiments.md` push 到 GitHub 并发给他
- 列 **3-5 个具体问题**（不要问"我该怎么做"，问"公式 21 在我的实现里 g 比 V 大很多是不是 reward scale 问题"）
- 把 `docs/research_context.md` 给他看 → 显示深读了论文
- 暑期跟 David 三方会议时用 `docs/open_questions_for_david.md` 作为提纲

---

## 六、容易踩的坑（已经踩过的，免得后人再踩）

### 训练相关
- ❌ **不要** 一开始就追求完整训练——先用 200K 步小实验验证 pipeline 对错
- ❌ **不要** 自己实现 HJ reachability 算法——论文用 RL 近似，按 SAC 改即可
- ❌ **不要** 改 `assetto_corsa_gym/AssettoCorsaPlugin/` 里的代码（运行在 AC 内嵌 Python，调试地狱）
- ❌ **不要** 让 10M_SAC 当 TRAINING driver（详见 [选项 2 dead end](daily_log.md#2026-05-20-晚)）
- ✅ **要** 把 γ_CBF 做成 config 参数（论文 Appendix D 做了 ablation）
- ✅ **要** 给 g(x) 做单元测试（出界 / 接近对手 / 正常行驶 三种状态手动 assert 符号）

### AC 平台坑（详见 `memory/reference_ac_pitfalls.md`）
- AC 加载插件路径分离仓库路径 → 建符号链接
- Race 模式 vJoy 输入设备被解绑 → 重启游戏前需手动 reload vJoy
- Race 模式 GT3 默认手动挡 → 需 Driving Aids 开自动挡
- OPP socket (2346) 在 daemon 线程 bind 失败但异常被吞 → 改用 MGMT 通道
- 10M_SAC 在 Race grid 起步严重 OOD（Hotlap 训的）

### 评估方法学坑（详见 `memory/reference_decoupled_eval.md`）
- 单模型评估对不会开车的 filter 是 spurious → 必须用解耦评估
- 先看 parquet 的 speed 列，车不动时 jerk/IM 都是幻觉
- 论文级数字必须报 mean ± std（n ≥ 3），单次结果会有 ±5 m/s² jerk 方差

### 工程教训
- 补丁链超过 3 层时主动退回评估（详见 `memory/feedback_retreat_on_compounding_patches.md`）
- 长训练用 `nohup`，但要确认 conda env 正确：`/home/wyb/anaconda3/envs/acgym/bin/python`
- 不要 mid-training 改 hyperparameter——记下来下个 run 改
