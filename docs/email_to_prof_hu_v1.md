# 给 Prof. Haimin Hu 的邮件草稿 v1

**创建时间：** 2026-05-17
**用途：** 复现 HCSF 论文 (RSS'25 "Safety with Agency") 过程中 3 个具体技术问题求教 + 算力受限下的方向建议
**详细决策依据：** 见 `~/.claude/plans/resilient-frolicking-swan.md`

---

## 发送前 checklist

发送前请逐项确认：

- [ ] 替换 `[Your Name]` 占位（你常用的英文落款，通常就是名字，比如 Yibo）
- [ ] 用学校邮箱发送（不要用 wybzkyy7932@gmail.com）
- [ ] 在已有的邮件 thread 里回复（保留主题/上下文）；如果新开邮件，用上面 Subject
- [ ] 决定是否**抄送 CQU 国内导师**（看你两位导师的沟通习惯）
- [ ] 发送时间：**北京时间周三 22:00 - 周四 0:00**（≈ EST 周三 9:00-11:00 上午）

---

## Subject

```
HCSF reproduction — three technical questions before summer
```

---

## Body

```
Dear Professor Hu,

As preparation for the summer, I have been reproducing HCSF on a personal
RTX 3060 setup, and now have the full Phase 0–5 pipeline running
end-to-end. A first 300K-step run on single-vehicle Barcelona, using the
released 10M-step pretrained SAC as the warmup driver, gave results that
align with H2 directionally:

    LRSF jerk = 2.2,  HCSF jerk = 1.2,  HCSF IM = 0.003

I would now like to push further toward the paper's setup. After
re-reading §V-C/D and Appendix C, I have three questions I would be very
grateful for your guidance on:

1. Reset-to-reference-path (§V-C-4): how was this implemented in AC? The
   plugin SDK does not seem to expose a setCarPosition-type API, so I
   suspect a plugin-level physics hook or a custom mod — could you share
   the approach (or relevant code if it's open)?

2. Number of training opponents (§V-C): what specific number did you use?

3. Warmup policies (§V-D, Eq. 17–20): were they trained separately and
   frozen, or jointly with the main HCSF networks? Roughly how many steps?

A note on my situation: with a single RTX 3060, a full 12.8M-step run is
out of reach. I would also greatly appreciate any suggestion on the most
valuable direction between now and July.

Thank you for your time.

Best regards,
[Your Name]
```

---

## 起草时的取舍说明（v5，备查）

| 选择 | 原因 |
|------|------|
| 英文 | Prof. Hu 在 JHU 任职，英文默认 |
| **引用 5/13 训练**（不引用 5/14 含对手训练）| 5/13 方向符合 H2；5/14 因结构性 gap 不诚实 |
| **主动披露用了 10M 预训练 SAC 作 warmup**（v5 新增）| 诚实交代实验设置；同时给 Q3 提供天然铺垫——"我用了 hack，你的正版怎么训"|
| 数字证据 1 行（jerk 2.2 vs 1.2, IM 0.003）| 一行结果说明问题 |
| **省略 Phase 0-5 详细 bullet** | 他知道 Phase 是什么 |
| **省略 V_φ 收敛值评论** | 他能从数据反推 |
| **Q1 加 "could you share the approach / code if open"**（v5 升级）| 不只问问题，主动 ask for resources，回复成本相同但收益翻倍 |
| 每个问题 1-2 行 | 信息密度高，快速扫读 |
| 算力受限放在最后 | 避免开头就显"求助" |
| 不提 GitHub repo / 5/14 对手集成 | 邮件极简 |
| 无附件 | 反垃圾过滤友好 |

---

## 预期回复路径（4 种情况的应对）

| 回复内容 | 你接下来做 |
|---------|---------|
| 给出 reset 实现方式（最理想） | Plan 第 4 步：实现 → 训 nominal + overtaking warmup → 跑 1M 步 HCSF 主训练 = 约一周 |
| "reset 是平台技巧，不公开" | 选 1：ego_server 协议自己扩 teleport 命令；选 2：放弃，写成 known gap |
| 建议转方向（如读他另一篇 paper / 换 topic） | 听他的，省 4-5 周做他建议的事 |
| 1 周不回复 | 发第二封短邮件（"想确认上一封是否收到"）；技术工作按"无 reset"假设继续 |

---

## 后续邮件（第二轮，W6 末）outline

待 W5-6 跑完训练后再起草，主要内容：
- 汇报第 2-5 步进展（含图表数据对比）
- 询问 W7 / 暑期具体研究方向预告
- 邀请他看 demo 视频
