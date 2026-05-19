# 给 David Oh 的技术问题清单

> **背景**：我（Yanbin Wang）2026 暑期来 Prof. Haimin Hu 实验室做 safe RL，已开始复现 HCSF。当前状态：Phase 0-5 pipeline 跑通，305K 步 single-vehicle Hotlap 训练完成（5/13），方向上符合 H2。
>
> Prof. Hu 5/18 邮件建议**赴 JHU 后跟你三方会议集中过技术问题**。本文档持续累积复现过程中的具体疑问，按优先级排序，开会前可逐条过。

---

## P1：影响复现路径的核心问题

### Q1：Reset-to-reference-path 实现细节（§V-C-4）

**背景**：论文写 "automatically reset to the closest point on the reference path"。AC plugin SDK 我没找到 `setCarPosition` API；`ac.ext_resetCar()` 只能回 pit/grid。5/14 尝试 Race grid 起步训练后严重 OOD——10M warmup driver 输出"小油门+刹车"，车几乎不动，305K 步训练里 buffer 99% 是 stationary 状态。

- **Q1.1** 这个 reset 你们具体是怎么实现的？plugin-level physics state write？AC 某个未文档化的 internal API？还是第三方 mod 配合？
- **Q1.2** Reset 时除位置外，车辆的 yaw / orientation 也对齐到 racing line 的切线方向吗？速度归零吗？
- **Q1.3** 这段实现代码 / plugin 是否可以共享给我（即便只是私下 reference）？

---

### Q2：Warmup policies 训练方式（§V-D, Eq. 17-20）

**背景**：Prof. Hu tentatively 说"frozen"，但他建议跟你 double-check。我现在用 Remonda et al. 2024 的 10M_SAC（单车 Hotlap 训）当 warmup driver 作为占位实现。

- **Q2.1** Nominal (Eq.17) 和 Overtaking (Eq.20) policies 各自训了多少步？训练环境跟主 HCSF 一致吗（同样 reset 机制、同样对手数量）？
- **Q2.2** 是先训完两个 policy 然后**完全 frozen** 用作 warmup driver，还是加载后跟主 HCSF networks **jointly fine-tuned**？
- **Q2.3** Table V 里 `c1=1/12, c2=300, c3=600, d^over_oppo=100m` 这些数字来源？做过 sensitivity ablation 吗？

---

### Q3：训练对手数量与配置（§V-C）

**背景**：Prof. Hu 暂时说"1 或 0 都行"，但 paper-level reproduction 想匹配你们的 setup。

- **Q3.1** 12.8M 步主训练用了几个对手？整个训练期间数量是固定的吗？
- **Q3.2** 对手都是 Mazda MX-5 ND 还是多种？AI strength=50%, aggression=30% 是所有对手统一吗？
- **Q3.3** Reset 时对手位置怎么决定（grid sequence？racing line 上某个偏移？跟自车的相对距离 sampled？）？

---

## P2：影响 V_φ / Q-CBF 行为的问题

### Q4：V_φ 训练动态

**背景**：我们 5/14 用 305K 步 Race-grid setup 训出来后做解耦评估（u^human=10M_SAC, filter=305K F1），HCSF 干预率 26.6%，但伴随大量 "no candidate satisfies Q-CBF, using fallback" → jerk 反而比 LRSF 高 2 倍（论文 H2 反向）。诊断为 V_φ 过度激进（OOD 训练所致）。

- **Q4.1** 你们训练**早期**（前 1-2M 步）V_φ 有类似的 over-aggressive / over-cautious 现象吗？
- **Q4.2** V_φ 跟 Q 协调通常需要多少步？训练时有什么 warning sign 说明 V_φ 学坏了？
- **Q4.3** V_φ 的 Bellman target（Eq.21）里 `max_u Q_φ(x, u)` 用的是 target Q 还是 online Q？

---

### Q5：Q-CBF 候选搜索失败模式（Algorithm 2）

**背景**：我们经常看到 `no candidate satisfies Q-CBF, using fallback`——尤其在高 noise 或 V_φ 偏激进时。

- **Q5.1** 你们训练完成模型在用户研究里这种 fallback 发生率大概是多少？
- **Q5.2** Fallback 时的实际动作是什么——纯 π^♦(x)，还是 u_human + π^♦ 的某种 blend？
- **Q5.3** 2000 candidates on the line between `u^human` and `u^♦` 这个数字怎么定的？有 ablation 吗？

---

## P3：评估方法学

### Q6：技术 ablation 的实施细节（Fig 15 等）

**背景**：你们 Fig 15 类型的 filter-only ablation 用 "overtaking policy + filter" vs "overtaking policy alone"。我做类似实验时改了 `evaluate_hcsf.py` 加了 `--human-model` / `--filter-model` 解耦参数。

- **Q6.1** 评估时给 u^human 加噪声吗？什么 noise model（高斯？OU process？）？σ 多少？
- **Q6.2** 每个 ablation 数据点跑多少步？多少 seed？停止条件是什么？

---

## P4：工程 / 实操

### Q7：长训练稳定性 + AC 平台坑

**背景**：12.8M 步约 3 周——Prof. Hu 说 RTX 3060 也能 1-2 周搞定。我们 RTX 3060 上目标 1M 步 ~14h。已踩过的坑：vJoy 切 Race 模式被重置、GT3 默认手动挡、OPP socket 2346 在 plugin daemon 线程 bind 静默失败、AC 加载插件路径分离于仓库路径需要符号链接。

- **Q7.1** 你们 12.8M 训练崩过吗？checkpoint + 自动 recovery 用了什么策略？
- **Q7.2** AC 长跑用 Hotlap 还是 Race 模式？怎么解决 Race 100 圈结束的问题？
- **Q7.3** 你们用了哪些 AC mod 或修改过的 plugin（除了 `sensors_par`）？

---

### Q8：vJoy / SCI 接口稳定性

**背景**：5/18 长训练前发现 vJoy 在 AC 切到 Quick Race 后会被解绑，需要在游戏内 reload。

- **Q8.1** vJoy 在你们 setup 里稳定吗？或者用了别的虚拟手柄（vGen, etc.）？
- **Q8.2** SCI 通信失败时（packages lost）怎么 detect + 恢复？

---

## P5：论文细节澄清

### Q9：关键公式 / 引用

- **Q9.1** 公式 23 policy gradient：`L(θ) = E[-Q_φ(x,u) + α log π_θ(u|x)]`。这里 u 用 reparametrization trick（gradient through sample）还是 stop-gradient？
- **Q9.2** 论文多处 "Following [19, 22]"（Fisac 2019 + Hsu 2023）。我已读 Hsu 2023 IS​AACS；[19] Bridging HJ + RL 还有必要精读吗，重点是哪几节？
- **Q9.3** Appendix C-1 提到的 init phase Q^init_term=2 阈值，你们对这个值做过 sensitivity test 吗？训练初期 Q 还没收敛时这个阈值怎么用？

---

## 持续更新机制

- **2026-05-18 初版**：5 天复现经验累积的 9 个问题
- 每次发现新问题追加到对应 P1-P5
- 赴 JHU 前 1 周再 review 一遍，把重复的合并、把已 self-resolve 的删除
- 开会时按优先级 P1 → P5 逐条过
