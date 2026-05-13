"""
以人为中心的安全滤波器 (Human-Centered Safety Filter, HCSF)

论文 §IV, 定义 2 (Eq. 11), 算法 2:
    每时间步求解 OCP:
        u*(x) = argmin_u  ||u_human(x) − u||²
                  s.t.    Q(x, u) ≥ γ·V(x)          (Q-CBF 约束, 命题 1)

求解方法 (Appendix C-4): 在 u_human → u^♦ 连线上采样 2000 候选,
选择满足 Q-CBF 且 ‖u − u_human‖² 最小的候选.

γ = 0.7 (Appendix C-5): 在能动性与舒适性之间取得平衡.

V(x) 来源:
  - v_net 提供时: V(x) = V_φ(x)  (Phase 2 训练的独立 V 网络)
  - v_net=None 时: V(x) = Q(x, π^♦(x))  (论文原始做法, §III-B)
"""

import torch
import numpy as np

import logging
logger = logging.getLogger(__name__)


class HCSF:
    """以人为中心的安全滤波器 (论文 Eq. 11, 算法 2)"""

    def __init__(self, q_net, policy_net, device,
                 v_net=None, gamma_cbf=0.7, n_candidates=2000):
        """
        Args:
            q_net:        双 Q 网络, 已训练
            policy_net:   回退策略 π^♦, 已训练
            device:       torch device
            v_net:        安全值网络 V_φ (可选). None 时从 Q 推导 V(x)
            gamma_cbf:    Q-CBF 设计参数 (论文 γ=0.7)
            n_candidates: 候选动作采样数 (论文 2000)
        """
        self._q_net = q_net
        self._v_net = v_net
        self._policy_net = policy_net
        self._device = device
        self._gamma_cbf = gamma_cbf
        self._n_candidates = n_candidates

        self._q_net.eval()
        self._policy_net.eval()
        if self._v_net is not None:
            self._v_net.eval()

        # 预计算 α ∈ [0, 1], 均匀分布在线段上
        self._alphas = torch.linspace(0.0, 1.0, n_candidates, device=device)

    def _compute_v(self, state_t):
        """V(x) = max_u Q(x,u)  论文 §III-B"""
        if self._v_net is not None:
            return self._v_net(state_t)
        else:
            _, _, u_fallback = self._policy_net(state_t)
            q1, q2 = self._q_net(state_t, u_fallback)
            return torch.min(q1, q2)

    def filter(self, state, human_action):
        """
        对单个人类动作进行 HCSF 安全滤波 (算法 2).

        Returns:
            filtered_action: numpy array [action_dim]
            info: dict with {intervened, v_value, im, n_satisfied, ...}
        """
        state_t = torch.tensor(
            state[None, ...], dtype=torch.float, device=self._device)
        u_human = torch.tensor(
            human_action[None, ...], dtype=torch.float, device=self._device)

        with torch.no_grad():
            # 1. V(x) 和回退动作 u^♦(x)
            v_x = self._compute_v(state_t)  # [1, 1]
            _, _, u_fallback = self._policy_net(state_t)  # [1, action_dim]

            # 2. 在 u_human → u_fallback 线段上采样候选
            delta = u_fallback - u_human
            candidates = u_human + self._alphas[:, None] * delta
            # candidates: [n_candidates, action_dim]

            # 3. Batch 评估 Q(x, candidate)
            n = self._n_candidates
            states_repeated = state_t.repeat(n, 1)
            qs1, qs2 = self._q_net(states_repeated, candidates)
            qs = torch.min(qs1, qs2)  # [n, 1]

            # 4. Q-CBF 约束: Q(x,u) ≥ γ·V(x)  (命题 1, Eq. 10)
            threshold = self._gamma_cbf * v_x
            satisfied = (qs >= threshold).squeeze()

            # 5. 选满足约束且偏离最小的候选
            if satisfied.any():
                dists = torch.sum((candidates - u_human) ** 2, dim=1)
                dists[~satisfied] = float('inf')
                best_idx = torch.argmin(dists)
                u_filtered = candidates[best_idx:best_idx+1]
                filtered_action = u_filtered.cpu().numpy()[0]
                intervened = not torch.allclose(u_human, u_filtered, atol=1e-4)
            else:
                # 不应发生 (u_fallback 即 α=1 总是满足 Q ≥ γV), 仅作保护
                logger.warning("HCSF: no candidate satisfies Q-CBF, using fallback")
                filtered_action = u_fallback.cpu().numpy()[0]
                intervened = True

        im = float(np.linalg.norm(human_action - filtered_action))

        info = {
            'intervened': intervened,
            'v_value': v_x.item(),
            'im': im,
            'human_action': human_action.copy(),
            'filtered_action': filtered_action.copy(),
            'fallback_action': u_fallback.cpu().numpy()[0],
            'n_satisfied': satisfied.sum().item(),
            'filter_type': 'HCSF',
        }

        return filtered_action, info

    def to(self, device):
        self._device = device
        self._alphas = self._alphas.to(device)
        self._q_net = self._q_net.to(device)
        self._policy_net = self._policy_net.to(device)
        if self._v_net is not None:
            self._v_net = self._v_net.to(device)
        return self
