# 每日进度日志

> 详细 commit 历史见 `git log`。本文档累积所有每日工作记录，CLAUDE.md 只保留最近 7 天滚动窗口。
> 实验细节（评估表、模型档案）见 `docs/experiments.md`。
> Phase 0-5 计划、工作流约定见 `docs/workflow_guide.md`。

---

## 2026-05-12 (Day 1) — Phase 0-2 基础设施

- Git 初始化 + wandb 配置 + 基线 SAC 15K 步验证
- **Phase 1**: g(x) 进入训练循环（ac_env / replay_buffer / sac / agent 全链路）
- **Phase 2**: V_φ 安全值网络（`algorithm/hcsf/safety_value.py`）
- Commit: `7daa018`
- **下一步**: Phase 3 运行时滤波器

---

## 2026-05-13 (Day 2) — Phase 3-5 + 方案 B 长训练

- **Phase 3**: LRSF / HCSF 滤波器（`lrsf_filter.py` + `hcsf_filter.py`）
- **Phase 4**: 训练课程（`warmup_policy.py` + `training_phases.py`）
- **Phase 5**: 评估脚本 `evaluate_hcsf.py`
- **300K 步 Hotlap 训练成功**（用 10M_SAC 当 warmup driver）:
  - 模型: `outputs/20260513_300K_hotlap_baseline/model/final/`（含 v_net）
  - 评估 (noise=0.3, 500 步): HCSF jerk=1.2 < LRSF jerk=2.2，HCSF IM=0.003（**符合 H2**）
  - V_φ 收敛到 ~4（论文期望 5-15，训练步数不够但方向对）
- **关键教训**: 训练 #1 失败因冷启动车不动；缩短 warmup 到 5s 后 buffer 效率从 2.4% → 58%
- **当前进度**: Phase 0-5 全部完成，端到端 pipeline 可运行。模型质量受限于训练步数。

---

## 2026-05-14 (Day 3) — 对手集成 + F1 训练 + 失败的方案 C++

### A-E 阶段：对手通路打通（约 2 小时）

- 修改：
  - `structures.py`: Opponent 加 `brakeStatus` 字段
  - `sensors_par.py`: 给 MGMT (2347) 加 `get_opponents` 命令绕过 OPP (2346) socket bind bug
  - `ac_client.py`: `SimulationManagement.get_opponents()`
  - `ac_env.py`: `g(x) = min(track_dist, opp_signed_dist - 5)`，新增 `enable_opponent` / `enable_opponent_in_obs` 开关
  - `scripts/probe_opponents.py` + `scripts/test_opponent_integration.py`
- 工程坑：
  - AC 加载的是 `~/.steam/.../apps/python/sensors_par/` 不是仓库版 → 建符号链接
  - Race 模式下 vJoy 输入设备被重置 → 重启游戏前需手动 reload vJoy
  - Race 模式 GT3 默认手动挡 → 需 Driving Aids 开自动挡
  - OPP socket (2346) 在 daemon 线程 bind 失败但异常被吞 → 改用 MGMT 通道
- 端到端 100 步测试通过

### F1 训练（305K 步，~4 小时）

- Quick Race 模式 + 1 Mazda + 10M_SAC warmup + 启用对手 g
- 模型: `outputs/20260514_305K_race_OODfail/model/final/`（含 v_net）
- 初次评估 HCSF jerk=46.7 > LRSF jerk=23.2（**反常**），当时误诊为"V_φ vs Q 不协调"
- 10M_SAC 在 Race grid 起步严重 OOD（Hotlap 训的）→ deterministic action 出"小油门+踩刹"

### 方案 C++ 失败（1.5 小时）

- 尝试：加载 10M 权重 + lr 1e-4 + deterministic patch + bootstrap 5 个补丁链
- 失败：补丁互相依赖，根本不可能修——**OOD 才是根因**（Race grid 起步对 10M Hotlap 训出的 policy 是 OOD）
- 教训进 memory: `feedback_retreat_on_compounding_patches.md`

---

## 2026-05-17 (Day 5) — 战略转向 + 邮件起草

- 制定赴 JHU 前 ~6.5 周路线图（`~/.claude/plans/resilient-frolicking-swan.md`）
- 深度阅读论文 §V-D + Appendix C，明确 3 个结构性 gap 与 paper 差距
- 起草邮件 v5 给 Prof. Hu：3 个具体技术问题 + 算力说明 + 求方向
- **战略转向**: 放弃数字复现，转向方法论复现 + 深度理解 + 与 Prof. Hu 技术对话
- **决策**: 第 1 步（发邮件）阻塞所有技术工作，第 2-4 步本周做不依赖回复的小任务

---

## 2026-05-18 (Day 6) — 解耦评估突破 + Prof. Hu 回复

### 任务 #1: 清理方案 C++ 残留（commit `46e8b25` → `3b2c558`）

- config.yml 关 `init_from_pretrained_path` / `bootstrap_steps` / lr 回 3e-4
- train.py 删除方案 C++ 加载块

### 任务 #2: 解耦评估方法学突破

- 修改 `evaluate_hcsf.py` 加 `--human-model` / `--filter-model`（论文 §VI / Appendix C-4 设计）
- **关键发现**：单模型评估对 5/14 F1 是 spurious 的——policy 不会开车 → 车不动 → jerk≈0 是算术幻觉
- 实测：当天上午 4 个 use_v_net run 里 9/12 episode speed mean < 0.01 m/s
- **解耦评估结果**（10M as human + F1 as filter, noise=0.3）见 `experiments.md`
- **方法学价值**：5/14 F1 模型真实行为首次被测出（之前所有评估都是 spurious）
- 新增 memory: `reference_decoupled_eval.md` + `feedback_distinguish_train_vs_eval_mobility.md`

### 邮件发送 + Prof. Hu 当天回复（关键转向）

详见 `memory/project_prof_hu_reply_20260518.md`。要点：
- Q1 Reset：暑期跟 David 当面解决，目前用现有 reset
- Q2 对手数：1 或 0 都 OK
- Q3 Warmup：tentatively frozen，David 确认
- **算力**: RTX 3060 跑 12.8M 可行（1-2 周），keep training
- **JHU 设备**: workstation + simulator 7 月第二周到
- **用户决定**: July 1 准时到 JHU，先用 laptop
- **战略转向**: 不再等 reset，按 Hu 建议直接长训练

### 任务 #3: 起草 `docs/open_questions_for_david.md`

- 9 个技术问题 / 25 个子项，分 P1-P5 优先级
- 暑期跟 David 三方会议时使用

### 清理工作

- 撤回 `docs/email_to_prof_hu_v1.md` 跟踪（已发完，commit `7c85c6a`）
- 优化 `.gitignore`（+65 行）

---

## 2026-05-19/20 — G2 1M Hotlap 长训练完成

- **1M Hotlap 长训练完成**: `outputs/20260519_G2_1M_plateau/model/final/`（含 v_net.pth）
  - 总步数 1,006,117，2022 episode，运行 ~12h
  - 技术路径: warmup(10M, 5s) → 跳过 init → 直接 training(SAC 探索)
  - **关键修改**: `agent.py:194` 跳过 init 阶段，init 的随机/对抗动作在 Hotlap 下导致频繁 crash，数据效率从 ~20/ep 提升到 ~100-150/ep
  - 4 次失败启动（nohup 缓冲 / conda run / init crash / 无 phases 冷启动），仅保留最终目录
- **训练曲线 plateau**: V_φ 从 2.1 升到 4.0（前 150K 步），之后 850K 步只涨到 4.77
- **解耦评估**: 1M 不比 300K 好（详见 `experiments.md`）
- **根因**: Hotlap 单车模式缺乏危险场景（g(x) 始终 > 0），V_φ Bellman 目标在安全区域退化为 Q 影子
- **当前进度**: 1M 模型产出但未超越 300K。RTX 3060 验证 12h 可跑 1M
- **下一步**: plan mode 规划下阶段——Race + 对手 + init，解决 OOD 问题

---

## 2026-05-20 晚 — G3/G4 失败 + Path C 规划 + G5 启动

- **G3 (612K kill)**: Hotlap 1M + INIT 恢复 + 4 项观测加固。π^♦ 驱动 TRAINING，speed 单调下降 20→13 m/s，63% episode "Speed too low" 终止。同 5/19 plateau 模式
- **G4 (122K kill, 选项2 dead end)**: 改 agent.py 让 10M_SAC 驱动 TRAINING + 噪声 0.1（误读 paper §V-D intent）。10M 太能开 → episode 跑满 15001 步 timeout → INIT ratio 从 5/19 的 1-4% 跌到 0.02% → V_φ stuck @ 2.4。**结构错误：10M 不会失败 → 永远不触发 episode reset → INIT 几乎不点火，回退**
- **路径决策**（重读论文 §V-C-4 + §V-D）:
  - 论文 episode 自然终止：`g<0` 或 stationary（无 step cap）
  - 论文 π^♦ 经常失败 → episode 短 → 频繁 reset → INIT ratio ~3%
  - 我们 1M 步预算下，**短 episode hack 可补偿 π^♦ 失败频率不足**
- **G5 启动 (Path C)**: π^♦ 驱动（paper-aligned 回退选项 1）+ `max_episode_py_time=60s`（从 600s 砍到 60s，强制 10x reset → 10x INIT）+ 1M 步，预计 ~11h 过夜跑完
- **GPU 瓶颈澄清**: 实测 3060 GPU 不满载——AC 物理仿真 ~30ms/step（CPU 为主）才是瓶颈，NN 训练只占 5-10ms/step。4090 比 3060 只快 10-15%。**Paper-scale 12.8M 在 3060 上 1-2 周完全可行**，印证 Hu 5/18 "keep training"

---

## 2026-05-21 — 🏆 G5 50K SOTA 发现 + Variance Eval + 大清理

### G5 训练完成 1M 步 + V_φ 发散

- 模型: `outputs/20260520_G5_1M_pathC_div_50KSOTA/`
- 训练曲线（V_φ 100K-130K 步发散）见 `experiments.md`
- 后 95% 步数在毒化网络上训练，浪费了

### 50K checkpoint 是迄今 SOTA

- 解耦评估初次：HCSF IM **0.0074**, jerk **59.9**（< LRSF 65.9 ✅ H2 满足），干预 **1.2%**
- 100K checkpoint 已退化（IM 0.42, 115/500 步车飞出）→ 证实 50K 是甜蜜点
- **关键结论**: Path C 短 episode hack 是正确方向。INIT ratio 起来后 V_φ 快速学到边界
- 唯一问题：V_φ 需要 target clipping + 早停防止发散

### Variance Eval（9 次复测 × 500 步）

- 完整 9 行数据 + 统计聚合见 `experiments.md`
- 核心结论：**G5 50K = 真 SOTA，不是偶然**
- IM 区间完全不重叠：G5 max 0.0179 < 5/13 min 0.0502
- 干预率不重叠：G5 max 1.8% < baseline min 4.8%
- 警告：jerk 单次方差大（±5），论文级图必须报 mean±std

### outputs/ 大清理 + 重命名

- 删除：G3 (709M, plateau 重复 G2)、G4 (261M, dead end)、G5 final/replay_buffer.pkl (8.2G, 毒化数据) → 共省 ~9G
- 4 个保留目录重命名为可读名（见 `experiments.md` 模型档案表）

### 文档大整理（5/21 晚）

- 新建 `docs/daily_log.md`、`docs/experiments.md`
- `docs/workflow_and_timeline.md` → `docs/workflow_guide.md`（精简）
- `CLAUDE.md` 进度日志只保留最近 7 天

---

## 2026-05-22 — G6 训练 + V_φ Fix 验证 + 15 次 Variance Eval

### G6 训练（Path C + V_φ Stability Fix）

V_φ Stability Fix 改动:
- `safety_value.py`: V_target 用 `target_q_net`（非 online Q）+ clamp [-30, 30] + gradient clip max_norm=10
- `sac.py`: 传递 `target_q_net` 给 v_trainer

**训练结果**:
- 模型: `outputs/20260521_204651.815/`（启动 5/21 20:46）
- 训练时长: 9h 26min，完成 **789K 步 (78.9%)**
- 终止原因: AC socket UTF-8 解码错误（**AC 平台 bug，不是我们代码**）
- **V_φ 完全没发散**：v_mean 在 [3.5, 4.9] 健康区间，对比 G5 同期 -2 → -18,228 完全反转
- v_mean 峰值: **4.90 @ step 549K**（接近 G2 1M plateau 4.77）
- 15 个 checkpoints 保存（50K → 750K）

详见 `experiments.md` Exp 7。

### G6 跨 timeline Variance Eval（15 次 × 500 步）

测试了 G6 5 个 checkpoint 各 3 次：50K / 100K / 200K / 549K / 750K

**关键发现：**

1. **G6 全 timeline 一致 "active filter"**:
   - 干预率稳定 11-16%（G5 50K 是 1.2% outlier）
   - IM 单调下降 0.162 → 0.089（V_φ 学得更准）

2. **G5 50K 真是 "happy accident"**：
   - G6 50K（同 step）IM 0.162 vs G5 50K 0.010 → 16× 差异
   - V_φ stability fix 改变了整个训练 trajectory
   - G5 50K 的 passive filter 不可复现

3. **G6 100K = 新"稳定 SOTA"**:
   - 3/3 次满足论文 H2（HCSF < LRSF jerk）—— **全部候选里 H2 满足率最高**
   - IM 0.127 ± 0.016（极小方差），jerk 61.07 ± 2.38（极小方差）

**两种 SOTA 叙事**:
- **数值最优**: G5 50K（outlier，IM 极低但不可复现）
- **稳定可复现**: G6 100K（H2 3/3 满足，方差最小）

完整 15 次原始数据 + 聚合表见 `experiments.md` "SOTA 实证 II" 节。

### 文档更新

- `experiments.md`: 新增 Exp 7 (G6 训练) + G6 跨 timeline variance eval（15 次原始数据 + 聚合）+ 完整 SOTA 排名
- `daily_log.md`: 本条 5/22 entry
- `CLAUDE.md`: 5/22 进度日志条目（V_φ fix 成功 + 两 SOTA）
