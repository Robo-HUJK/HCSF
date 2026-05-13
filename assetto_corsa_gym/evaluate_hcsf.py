#!/usr/bin/env python3
"""
Phase 5: HCSF 评估脚本

对比 HCSF / LRSF / None 三种安全滤波模式在 AC 环境中的表现.
记录三个论文指标:
  - IM  (Input Modification,  Eq. 12): ||u_human - φ(x, u_human)||₂
  - Jerk (                    Eq. 13): ||p̈||₂  (位置三阶导数)
  - ID  (Input Difference,    Eq. 14): ||φ_t - φ_{t-1}||₂²

用法:
  python evaluate_hcsf.py --model <path> [--steps 500] [--noise 0.3]

模型来源:
  - 方案 A: 预训练 10M SAC (V 从 Q 推导)
  - 方案 B: Phase 1-4 完整训练模型 (V 从 V_φ 网络)
"""

import sys, os
sys.path.extend([
    os.path.abspath('./assetto_corsa_gym'),
    './algorithm/discor',
    './algorithm',
])

import argparse
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from discor.network import TwinnedStateActionFunction, GaussianPolicy
from hcsf.safety_value import SafetyValueNetwork
from hcsf.lrsf_filter import LRSF
from hcsf.hcsf_filter import HCSF
import AssettoCorsaEnv.assettoCorsa as assettoCorsa


class MetricsTracker:
    """IM / Jerk / ID 指标追踪器"""

    def __init__(self):
        self.records = []
        self._prev_filtered_action = None
        self._prev_velocity = None
        self._prev_accel = None
        self._dt = 1.0 / 30.0  # 30 Hz control

    def step(self, human_action, filtered_action, position, env_velocity):
        """
        记录一个时间步的指标.

        Args:
            human_action:   [3] 原始人类控制
            filtered_action:[3] 滤波后控制
            position:       [3] 车辆 3D 位置 (x, y, z)
            env_velocity:   [3] 车辆速度 (vx, vy, vz)
        Returns:
            metrics dict
        """
        im = float(np.linalg.norm(human_action - filtered_action))  # Eq. 12

        # Jerk (Eq. 13): acceleration of acceleration
        velocity = np.array(env_velocity[:3], dtype=np.float64)
        accel = (velocity - self._prev_velocity) / self._dt if self._prev_velocity is not None else np.zeros(3)
        jerk = float(np.linalg.norm(
            (accel - self._prev_accel) / self._dt if self._prev_accel is not None else np.zeros(3)
        ))  # Eq. 13

        # ID (Eq. 14): squared difference of consecutive filtered actions
        if self._prev_filtered_action is not None:
            id_val = float(np.linalg.norm(filtered_action - self._prev_filtered_action) ** 2)
        else:
            id_val = 0.0

        self._prev_filtered_action = filtered_action.copy()
        self._prev_velocity = velocity.copy()
        self._prev_accel = accel.copy()

        return {'im': im, 'jerk': jerk, 'id': id_val}


def load_model(model_dir, state_dim, action_dim, device, use_v_net=False):
    """加载训练好的网络"""
    q_net = TwinnedStateActionFunction(
        state_dim, action_dim, [256, 256, 256]).to(device)
    q_net.load(f'{model_dir}/online_q_net.pth')
    q_net.eval()

    policy_net = GaussianPolicy(
        state_dim, action_dim, [256, 256, 256]).to(device)
    policy_net.load(f'{model_dir}/policy_net.pth')
    policy_net.eval()

    v_net = None
    if use_v_net:
        v_net = SafetyValueNetwork(state_dim, [256, 256, 256]).to(device)
        v_net.load(f'{model_dir}/v_net.pth')
        v_net.eval()

    return q_net, policy_net, v_net


def run_trial(env, filter_obj, policy_net, name, max_steps, noise_std, device):
    """运行一次评估试验"""
    tracker = MetricsTracker()
    state = env.reset()
    ep_return = 0.0
    intervened_steps = 0

    for step in range(max_steps):
        # 模拟人类: 策略确定性动作 + 高斯噪声
        state_t = torch.tensor(state[None], dtype=torch.float, device=device)
        with torch.no_grad():
            _, _, det_action = policy_net(state_t)
        human_action = det_action.cpu().numpy()[0]
        if noise_std > 0:
            human_action = human_action + np.random.normal(0, noise_std, 3)
            human_action = np.clip(human_action, -1, 1)

        # 应用滤波器
        if filter_obj is not None:
            filtered_action, info = filter_obj.filter(state, human_action)
        else:
            filtered_action = human_action
            info = {'intervened': False, 'v_value': 0.0, 'im': 0.0}

        # 施加动作
        env.set_actions(filtered_action)
        next_state, reward, done, env_info = env.step(action=None)

        # 记录指标 — 位置/速度来自 env.state (expand_state 后的完整 telemetry)
        env_state = env.state
        pos = np.array([
            env_state.get('world_position_x', 0),
            env_state.get('world_position_y', 0),
            env_state.get('world_position_z', 0)], dtype=np.float64)

        vel = np.array([
            env_state.get('velocity_x', 0),
            env_state.get('velocity_y', 0),
            env_state.get('velocity_z', 0)], dtype=np.float64)

        metrics = tracker.step(human_action, filtered_action, pos, vel)
        metrics.update({
            'step': step,
            'v_value': info.get('v_value', 0.0),
            'intervened': int(info.get('intervened', False)),
            'reward': reward,
            'filter': name,
        })
        tracker.records.append(metrics)

        if info.get('intervened', False):
            intervened_steps += 1
        ep_return += reward
        state = next_state

        if done:
            break

    return tracker.records, ep_return, intervened_steps


def print_summary(name, records):
    """打印评估摘要"""
    df = pd.DataFrame(records)
    n = len(df)
    intv_pct = df['intervened'].mean() * 100 if 'intervened' in df else 0

    print(f'  [{name:5s}] steps={n:4d}  '
          f'IM_avg={df["im"].mean():.4f}  '
          f'jerk_avg={df["jerk"].mean():.1f}  '
          f'ID_avg={df["id"].mean():.4f}  '
          f'intervened={df["intervened"].sum():.0f}/{n} ({intv_pct:.1f}%)  '
          f'V_avg={df["v_value"].mean():.1f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                        default='/home/wyb/car/AssettoCorsaGymDataSet/data_sets/ks_barcelona-layout_gp/bmw_z4_gt3/20240404_SAC_10M/model/final')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--use-v-net', action='store_true')
    args = parser.parse_args()

    device = 'cuda'
    state_dim, action_dim = 125, 3

    config = OmegaConf.load('config.yml')
    q_net, policy_net, v_net = load_model(
        args.model, state_dim, action_dim, device, args.use_v_net)
    print('Model loaded OK')

    # 创建滤波器 (v_net=None 时 V 从 Q 推导)
    lrsf = LRSF(q_net, policy_net, device, v_net=v_net)
    hcsf = HCSF(q_net, policy_net, device, v_net=v_net)

    env = assettoCorsa.make_ac_env(cfg=config, work_dir='/tmp/ac_eval_hcsf')

    print(f'\n=== HCSF Evaluation (noise={args.noise}, steps={args.steps}) ===')
    print(f'Filter        | Steps | IM_avg  | Jerk_avg | ID_avg   | Intervened | V_avg')
    print('-' * 85)

    all_results = {}
    for filter_obj, name in [(None, 'None'), (lrsf, 'LRSF'), (hcsf, 'HCSF')]:
        records, ep_return, intv = run_trial(
            env, filter_obj, policy_net, name, args.steps, args.noise, device)
        df = pd.DataFrame(records)
        print_summary(name, records)
        all_results[name] = df

    env.close()

    # 对比摘要
    print('\n=== Comparison ===')
    for name in ['None', 'LRSF', 'HCSF']:
        df = all_results[name]
        print(f'  {name:5s}: IM_mean={df["im"].mean():.4f}, '
              f'jerk_mean={df["jerk"].mean():.1f}, '
              f'ID_mean={df["id"].mean():.4f}')

    # 保存
    combined = pd.concat(all_results.values(), keys=all_results.keys(), names=['filter'])
    csv_path = '/tmp/ac_eval_hcsf/metrics.csv'
    combined.to_csv(csv_path)
    print(f'\nSaved to {csv_path}')


if __name__ == '__main__':
    main()
