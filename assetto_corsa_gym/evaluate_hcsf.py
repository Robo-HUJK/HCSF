#!/usr/bin/env python3
"""
Phase 5: HCSF 评估脚本（支持解耦评估）

对比 HCSF / LRSF / None 三种安全滤波模式在 AC 环境中的表现.
记录三个论文指标:
  - IM  (Input Modification,  Eq. 12): ||u_human - φ(x, u_human)||₂
  - Jerk (                    Eq. 13): ||p̈||₂  (位置三阶导数)
  - ID  (Input Difference,    Eq. 14): ||φ_t - φ_{t-1}||₂²

设计 (论文 §VI / Appendix C-4):
  u^human(x)（动作输入源）和 filter 用的 Q+V+π^♦ 在论文里**天然解耦**:
    - 论文用户研究: u^human = 83 个真人方向盘输入; filter = 训出的 HCSF Q+V_φ
    - 论文技术 ablation (Fig 15): u^human = overtaking policy; filter 同 HCSF
  本脚本支持两种模式:
    - UNIFIED (单模型):  human 和 filter 都来自同一个 model（向后兼容）
    - DECOUPLED (解耦): human 来自 --human-model；filter 来自 --filter-model
                       推荐用 10M 预训练 SAC 当 human（保证车会开），
                       Phase 1-4 训出的模型当 filter（测试其 Q+V_φ 行为）

用法:
  # 解耦评估（推荐）：10M 当 human，5/14 F1 当 filter
  python evaluate_hcsf.py \\
      --human-model  /path/to/20240404_SAC_10M/model/final \\
      --filter-model /path/to/outputs/.../model/final \\
      --use-v-net --steps 300 --noise 0.3

  # 单模型评估（向后兼容）
  python evaluate_hcsf.py --model <path> --use-v-net --steps 300 --noise 0.3
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


def run_trial(env, filter_obj, human_policy_net, name, max_steps, noise_std, device):
    """运行一次评估试验.

    Args:
        env: AC 环境
        filter_obj: None / LRSF / HCSF 实例（filter 用的 Q+V+π^♦ 已 baked-in）
        human_policy_net: 用来模拟 u^human(x) 的策略网络
                         （解耦模式下推荐用 10M_SAC；单模型下=filter 同一个）
    """
    tracker = MetricsTracker()
    state = env.reset()
    ep_return = 0.0
    intervened_steps = 0

    for step in range(max_steps):
        # 模拟 u^human(x): human_policy_net 的确定性动作 + 高斯噪声
        # 这里 human_policy_net 可以是 10M 预训练 SAC（解耦）也可以是 filter 同源（单模型）
        state_t = torch.tensor(state[None], dtype=torch.float, device=device)
        with torch.no_grad():
            _, _, det_action = human_policy_net(state_t)
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
    # 单模型模式（向后兼容）：human 和 filter 都从这个路径加载
    parser.add_argument('--model', type=str,
                        default='/home/wyb/car/AssettoCorsaGymDataSet/data_sets/ks_barcelona-layout_gp/bmw_z4_gt3/20240404_SAC_10M/model/final',
                        help='单模型模式：human 和 filter 都从这里加载')
    # 解耦模式：human 和 filter 分别指定
    parser.add_argument('--human-model', type=str, default=None,
                        help='u^human(x) 的来源策略 (default: --model)。推荐 10M_SAC 保证会开车')
    parser.add_argument('--filter-model', type=str, default=None,
                        help='filter 的 Q+V+π^♦ 来源 (default: --model)。Phase 1-4 训出的模型')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--noise', type=float, default=0.3)
    parser.add_argument('--use-v-net', action='store_true',
                        help='从 --filter-model 加载独立 v_net.pth（否则 V 从 Q 推导）')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='输出 csv 文件名前缀，便于区分多次 run')
    args = parser.parse_args()

    # 解析路径：human 和 filter 默认都用 --model（单模型模式）
    human_path = args.human_model if args.human_model else args.model
    filter_path = args.filter_model if args.filter_model else args.model
    decoupled = (human_path != filter_path)

    device = 'cuda'
    state_dim, action_dim = 125, 3

    config = OmegaConf.load('config.yml')

    # 加载 human policy（只用 policy_net；不需要 q_net / v_net）
    print(f'[Loading HUMAN policy] {human_path}')
    _, human_policy_net, _ = load_model(
        human_path, state_dim, action_dim, device, use_v_net=False)

    # 加载 filter components（q_net + π^♦ fallback policy + 可选 v_net）
    print(f'[Loading FILTER] {filter_path}  (use_v_net={args.use_v_net})')
    filter_q_net, filter_policy_net, filter_v_net = load_model(
        filter_path, state_dim, action_dim, device, args.use_v_net)

    mode_label = 'DECOUPLED (论文式)' if decoupled else 'UNIFIED (单模型)'
    print(f'[Mode] {mode_label}')

    # 创建滤波器：用 filter_model 的 q/v/policy
    lrsf = LRSF(filter_q_net, filter_policy_net, device, v_net=filter_v_net)
    hcsf = HCSF(filter_q_net, filter_policy_net, device, v_net=filter_v_net)

    env = assettoCorsa.make_ac_env(cfg=config, work_dir='/tmp/ac_eval_hcsf')

    print(f'\n=== HCSF Evaluation (noise={args.noise}, steps={args.steps}, mode={mode_label}) ===')
    print(f'Filter        | Steps | IM_avg  | Jerk_avg | ID_avg   | Intervened | V_avg')
    print('-' * 85)

    all_results = {}
    for filter_obj, name in [(None, 'None'), (lrsf, 'LRSF'), (hcsf, 'HCSF')]:
        records, ep_return, intv = run_trial(
            env, filter_obj, human_policy_net,
            name, args.steps, args.noise, device)
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

    # 保存（前缀让多次 run 不互相覆盖）
    combined = pd.concat(all_results.values(), keys=all_results.keys(), names=['filter'])
    csv_name = f'{args.save_prefix}metrics.csv' if args.save_prefix else 'metrics.csv'
    csv_path = f'/tmp/ac_eval_hcsf/{csv_name}'
    combined.to_csv(csv_path)
    print(f'\nSaved to {csv_path}')


if __name__ == '__main__':
    main()
