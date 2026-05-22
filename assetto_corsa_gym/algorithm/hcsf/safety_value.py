import torch
from torch import nn
from torch.optim import Adam

from discor.network import BaseNetwork, create_linear_network

import logging
logger = logging.getLogger(__name__)


class SafetyValueNetwork(BaseNetwork):
    """
    安全值函数 V_φ(x): state → V(x) ∈ ℝ

    架构与 Q 网络一致 (3层 MLP, 每层 256 神经元), 但输入仅为 state.
    论文 §V-A: V(x) = max_u Q(x,u) 的独立近似, 用公式 21 训练.
    """

    def __init__(self, state_dim, hidden_units=(256, 256, 256)):
        super().__init__()
        self.net = create_linear_network(
            input_dim=state_dim,
            output_dim=1,
            hidden_units=list(hidden_units))

    def forward(self, states):
        return self.net(states)


class SafetyValueTrainer:
    """
    V_φ(x) 训练器.

    训练目标 (论文公式 21):
        V(x) ≈ (1 − γ_ENV)·g(x) + γ_ENV·min{ g(x), Q_φ(x, π(x)) }

    损失: L_V = E[ (V_φ(x) − V_target(x))² ]
    """

    def __init__(self, state_dim, device, gamma_env=0.992,
                 hidden_units=(256, 256, 256), lr=3e-4,
                 target_clip=30.0, grad_clip=10.0):
        """
        Args:
            target_clip: V_target clamp 范围 [-target_clip, target_clip]
                        g 函数典型值域 [-10, 15]，30 是宽松上界（5/21 G5 发散修复）
            grad_clip: V_φ 梯度 L2 norm 上限（防梯度爆炸）
        """
        self._device = device
        self._gamma_env = gamma_env
        self._target_clip = target_clip
        self._grad_clip = grad_clip

        self._v_net = SafetyValueNetwork(
            state_dim=state_dim, hidden_units=hidden_units).to(device)
        self._optim = Adam(self._v_net.parameters(), lr=lr)

        self._learning_steps = 0

    def compute_v_target(self, states, margins, target_q_net, policy_net):
        """
        计算 V_target = (1−γ)g + γ·min{g, Q_target(x, π(x))}

        关键修复 (5/21 G5 发散后):
        - 用 **target_q_net** 而非 online Q net 提供稳定 bootstrap
          (对齐 sac.py:190 的 Q-target 用 target_q_net 的设计)
        - V_target clamp 到 [-target_clip, target_clip] 防止极端 Q 值
          引发 V_φ 网络回归到不可达区域 (论文未明示，工程稳定补丁)
        """
        with torch.no_grad():
            actions, _, _ = policy_net(states)
            qs1, qs2 = target_q_net(states, actions)  # ← target net
            qs = torch.min(qs1, qs2)

            gamma_env = self._gamma_env
            v_target = (1.0 - gamma_env) * margins \
                + gamma_env * torch.min(margins, qs)

            # Clamp 防发散
            v_target = torch.clamp(v_target,
                                    min=-self._target_clip,
                                    max=self._target_clip)

        return v_target

    def update(self, states, margins, target_q_net, policy_net, writer=None):
        """
        更新 V 网络, 返回 V_loss 和 V_mean.
        """
        self._learning_steps += 1

        v_target = self.compute_v_target(states, margins, target_q_net, policy_net)
        v_pred = self._v_net(states)

        v_loss = torch.mean((v_pred - v_target).pow(2))

        self._optim.zero_grad()
        v_loss.backward()
        # Gradient clipping 防爆炸 (5/21 G5 发散修复)
        torch.nn.utils.clip_grad_norm_(self._v_net.parameters(),
                                        max_norm=self._grad_clip)
        self._optim.step()

        if writer is not None and self._learning_steps % 10 == 0:
            writer.add_scalar('loss/V', v_loss.detach().item(),
                              self._learning_steps)
            writer.add_scalar('stats/mean_V', v_pred.detach().mean().item(),
                              self._learning_steps)
            writer.add_scalar('stats/mean_V_target',
                              v_target.mean().item(),
                              self._learning_steps)

        return v_loss.detach().item(), v_pred.detach().mean().item()

    def v_net(self):
        return self._v_net

    def save_models(self, save_dir):
        self._v_net.save(f"{save_dir}/v_net.pth")

    def load_models(self, load_dir):
        self._v_net.load(f"{load_dir}/v_net.pth")
