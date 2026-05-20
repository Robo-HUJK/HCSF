"""
多阶段训练课程 (论文 §V-D, Appendix C-1, Table V)

每 episode 的三个阶段:
  1. Warmup:   用名义/超车策略将车加速到现实速度 (≤25s)
  2. Init:     对抗/随机/混合策略将车推入易失败状态
  3. Training: 标准 SAC 训练 (Phase 1-2 已实现)
"""

import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)

# 阶段常量
PHASE_WARMUP = 'warmup'
PHASE_INIT = 'init'
PHASE_TRAINING = 'training'

# 初始化策略类型
INIT_ADV = 'adversarial'
INIT_RAND = 'random'
INIT_MIX = 'mixed'


class TrainingPhases:
    """
    管理三阶段训练流程.

    用法 (在 train_episode 中):
        phases = TrainingPhases(policy_net, q_net, device, warmup_cfg, init_cfg)
        state = env.reset()
        phase = PHASE_WARMUP

        while not done:
            action, new_phase = phases.select_action(state, phase, step)
            phase = new_phase if new_phase else phase

            env.set_actions(action)
            next_state, reward, done, info = env.step()

            q_val = phases.get_q(state, action) if phase == PHASE_INIT else None
            if phases.should_end_phase(phase, info, q_val, step):
                phase = PHASE_INIT if phase == PHASE_WARMUP else PHASE_TRAINING

            state = next_state
    """

    def __init__(self, policy_net, q_net, device,
                 warmup_cfg=None, init_cfg=None, warmup_policy=None):
        self._policy_net = policy_net  # SAC 训练中的策略 (init 阶段用)
        self._q_net = q_net
        self._device = device
        self._warmup_policy = warmup_policy  # 预训练策略 (warmup 阶段用, None→使用 policy_net)

        # warmup 配置 (论文 Table V)
        self._warmup_cfg = warmup_cfg or {}
        self._T_warmup_max = self._warmup_cfg.get('T_warmup_max', 5)
        self._P_over = self._warmup_cfg.get('P_over', 0.6)
        # 对手相关 (暂未启用)
        self._P_oppo = self._warmup_cfg.get('P_oppo', 0.25)
        self._doppo_min = self._warmup_cfg.get('doppo_min', 6.0)
        self._doppo_max = self._warmup_cfg.get('doppo_max', 36.0)
        self._P_brake_epi = self._warmup_cfg.get('P_brake_epi', 0.4)
        self._P_brake_step = self._warmup_cfg.get('P_brake_step', 0.25)
        self._u_brake = self._warmup_cfg.get('u_brake', 0.6)
        self._v_warmup = self._warmup_cfg.get('v_warmup', 40.0)

        # init 配置 (论文 Table V)
        self._init_cfg = init_cfg or {}
        self._P_term = self._init_cfg.get('P_term', 0.2)
        self._Q_init_term = self._init_cfg.get('Q_init_term', 2.0)
        self._T_init_max = self._init_cfg.get('T_init_max', 3.0)  # 秒，硬上限
        self._P_FT = self._init_cfg.get('P_FT', 0.4)
        self._P_adv = self._init_cfg.get('P_adv', 0.3)
        self._P_rand = self._init_cfg.get('P_rand', 0.3)
        self._P_mix = self._init_cfg.get('P_mix', 0.4)

        # 阶段内状态
        self._phase = PHASE_WARMUP
        self._phase_start_step = 0
        self._init_substep = 0  # INIT 阶段内步数计数（重置于 start_phase）
        self._init_strategy = None  # adversarial/random/mixed
        self._init_ft_enabled = False
        self._warmup_brake_enabled = False
        self._warmup_use_policy = 'nominal'  # nominal or overtaking
        self._mix_use_adv = True  # mixed strategy alternation flag

    @property
    def phase(self):
        return self._phase

    def select_action(self, state, phase, current_step, action_dim=3):
        """
        根据当前阶段选择动作.

        Returns:
            action: numpy array [action_dim]
            new_phase: None (不变) 或新阶段名 (自动推进)
        """
        self._phase = phase
        state_t = torch.tensor(state[None, ...], dtype=torch.float, device=self._device)

        if phase == PHASE_WARMUP:
            return self._warmup_action(state_t), None

        elif phase == PHASE_INIT:
            return self._init_action(state_t, action_dim), None

        else:  # PHASE_TRAINING
            return None, None  # 由调用方用 SAC explore/exploit

    def _warmup_action(self, state_t, current_step=0):
        """warmup 阶段: 用预训练策略 (或 SAC 策略作为 fallback)"""
        pol = self._warmup_policy if self._warmup_policy is not None else self._policy_net
        with torch.no_grad():
            _, _, det_action = pol(state_t)
        action = det_action.cpu().numpy()[0]
        return action

    def _init_action(self, state_t, action_dim):
        """初始化阶段: 对抗/随机/混合 动作选择"""
        if self._init_strategy is None:
            self._pick_init_strategy()

        if self._init_strategy == INIT_ADV:
            return self._adversarial_action(state_t, action_dim)
        elif self._init_strategy == INIT_RAND:
            return self._random_action(action_dim)
        else:  # INIT_MIX
            self._mix_use_adv = not self._mix_use_adv
            if self._mix_use_adv:
                return self._adversarial_action(state_t, action_dim)
            else:
                return self._random_action(action_dim)

    def _adversarial_action(self, state_t, action_dim, n_candidates=500):
        """对抗初始化: 选 Q 最小的候选动作 (最危险的)"""
        with torch.no_grad():
            # 随机采样候选
            candidates = torch.rand(n_candidates, action_dim, device=self._device) * 2 - 1
            if self._init_ft_enabled:
                # 满油门: throttle=1, brake=-1
                candidates[:, 1] = 1.0
                candidates[:, 2] = -1.0

            states_tiled = state_t.repeat(n_candidates, 1)
            qs1, qs2 = self._q_net(states_tiled, candidates)
            qs = torch.min(qs1, qs2)
            worst_idx = torch.argmin(qs)
            action = candidates[worst_idx].cpu().numpy()
        return action

    def _random_action(self, action_dim):
        """随机初始化: 随机采样动作"""
        action = np.random.uniform(-1, 1, action_dim)
        if self._init_ft_enabled:
            action[1] = 1.0   # full throttle
            action[2] = -1.0  # no brake (actually max brake goes from -1 to 1)
            # 论文 U_FT: throttle=1, brake=-1 (in [-1,1] action space)
        return action

    def _pick_init_strategy(self):
        """按概率选择初始化策略"""
        r = np.random.random()
        if r < self._P_adv:
            self._init_strategy = INIT_ADV
        elif r < self._P_adv + self._P_rand:
            self._init_strategy = INIT_RAND
        else:
            self._init_strategy = INIT_MIX

        self._init_ft_enabled = np.random.random() < self._P_FT

    def get_q(self, state, action):
        """获取 Q(x,u) 用于 init 阶段终止判断"""
        state_t = torch.tensor(state[None, ...], dtype=torch.float, device=self._device)
        action_t = torch.tensor(action[None, ...], dtype=torch.float, device=self._device)
        with torch.no_grad():
            q1, q2 = self._q_net(state_t, action_t)
            q = torch.min(q1, q2)
        return q.item()

    def should_end_phase(self, phase, env_info, q_value, step):
        """
        判断是否结束当前阶段.
        返回 True → 调用方推进到下一阶段.
        """
        if phase == PHASE_WARMUP:
            return self._should_end_warmup(env_info, step)
        elif phase == PHASE_INIT:
            self._init_substep += 1
            return self._should_end_init(q_value, self._init_substep)
        else:
            return False

    def _should_end_warmup(self, env_info, step):
        """warmup 提前终止条件 (论文 Appendix C-1)"""
        # 时间到了
        if step >= self._T_warmup_max * 30:  # approx steps at 30Hz
            return True

        # 重刹车终止 (未启用: 需要额外环境信息)
        # if self._warmup_brake_enabled and env_info.get('brake_input', 0) > self._u_brake:
        #    if np.random.random() < self._P_brake_step:
        #        return True

        return False

    def _should_end_init(self, q_value, current_step=0):
        """初始化阶段终止: 超时 或 Q < Q_init_term 时以概率 P_term 结束"""
        limit = int(self._T_init_max * 25)  # 25Hz
        if current_step >= limit:
            logger.info(f"[Phase] INIT ended after {current_step} substeps (reason: timeout {self._T_init_max}s)")
            return True
        if q_value is not None and q_value < self._Q_init_term:
            if np.random.random() < self._P_term:
                logger.info(f"[Phase] INIT ended after {current_step} substeps (reason: Q={q_value:.3f}<{self._Q_init_term})")
                return True
        return False

    def start_phase(self, phase, current_step):
        """初始化阶段状态"""
        self._phase = phase
        self._phase_start_step = current_step
        logger.info(f"[Phase] step={current_step} entering {phase}")

        if phase == PHASE_WARMUP:
            self._warmup_brake_enabled = np.random.random() < self._P_brake_epi
            self._warmup_use_policy = 'overtaking' if np.random.random() < self._P_over else 'nominal'

        elif phase == PHASE_INIT:
            self._init_substep = 0
            self._init_strategy = None  # 首次 select_action 时选择策略
            self._init_ft_enabled = False
