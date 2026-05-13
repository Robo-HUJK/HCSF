"""
最后一刻安全滤波器 (Last-Resort Safety Filter, LRSF)

论文 §III-B, 公式 7:
    u(x) = π^task(x)       if V(x) > 0
           π^♦(x)          otherwise (V(x) = 0)

在安全边界处完全切换到回退策略，无视人类驾驶员意图。
作为 HCSF 的对比基线。

V(x) 来源:
  - v_net 提供时: V(x) = V_φ(x)  (Phase 2 训练的独立 V 网络)
  - v_net=None 时: V(x) = Q(x, π^♦(x))  (论文原始做法, §III-B)
π^♦(x) = argmax_u Q(x,u) ≈ π_θ(x) (确定性模式)
"""

import torch

import logging
logger = logging.getLogger(__name__)


class LRSF:
    """最后一刻安全滤波器 (论文 Eq. 7)"""

    def __init__(self, q_net, policy_net, device, v_net=None):
        """
        Args:
            q_net:      双 Q 网络, 已训练
            policy_net: 回退策略 π^♦, 已训练
            device:     torch device
            v_net:      安全值网络 V_φ (可选). None 时从 Q 推导 V
        """
        self._q_net = q_net
        self._v_net = v_net
        self._policy_net = policy_net
        self._device = device

        self._q_net.eval()
        self._policy_net.eval()
        if self._v_net is not None:
            self._v_net.eval()

    def _compute_v(self, state_t):
        """V(x) = max_u Q(x,u)  论文 §III-B, Eq. 4-5"""
        if self._v_net is not None:
            return self._v_net(state_t)
        else:
            _, _, u_fallback = self._policy_net(state_t)
            q1, q2 = self._q_net(state_t, u_fallback)
            return torch.min(q1, q2)

    def filter(self, state, human_action):
        """
        对单个人类动作进行安全滤波.

        Returns:
            filtered_action: numpy array [action_dim]
            info: dict with {intervened, v_value, filter_type, ...}
        """
        state_t = torch.tensor(
            state[None, ...], dtype=torch.float, device=self._device)

        with torch.no_grad():
            v = self._compute_v(state_t).item()

        if v > 0:
            return human_action, {
                'intervened': False,
                'v_value': v,
                'filter_type': 'LRSF',
            }
        else:
            with torch.no_grad():
                _, _, fallback = self._policy_net(state_t)

            fallback_action = fallback.cpu().numpy()[0]
            return fallback_action, {
                'intervened': True,
                'v_value': v,
                'human_action': human_action.copy(),
                'fallback_action': fallback_action,
                'filter_type': 'LRSF',
            }

    def to(self, device):
        self._device = device
        self._q_net = self._q_net.to(device)
        self._policy_net = self._policy_net.to(device)
        if self._v_net is not None:
            self._v_net = self._v_net.to(device)
        return self
