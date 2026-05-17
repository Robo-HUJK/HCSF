"""
阶段 E 端到端验证 - 走法 2（保守版）
跑 ~100 步，验证：
  1. 环境能在 enable_opponent=True 下初始化
  2. state['opp_signed_dist'] 字段存在
  3. state['margin'] = min(track, opp) 正确
  4. obs.shape 比原版多 1 维（如果 enable_opponent_in_obs=True）
  5. 自车移动 / 对手位置变化时 opp_signed_dist 跟着变
"""
import sys
import os
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "assetto_corsa_gym"))
sys.path.insert(0, REPO_ROOT)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

import numpy as np
from omegaconf import OmegaConf
import AssettoCorsaEnv.assettoCorsa as assettoCorsa
make_ac_env = assettoCorsa.make_ac_env

CFG_PATH = os.path.join(REPO_ROOT, "config.yml")
N_STEPS = 100


def main():
    cfg = OmegaConf.load(CFG_PATH)
    # 强制开启对手集成（不改 config.yml，仅测试本次）
    cfg.AssettoCorsa.enable_opponent = True
    cfg.AssettoCorsa.enable_opponent_in_obs = True
    # 关掉训练课程，免去 warmup 模型加载
    cfg.AssettoCorsa.enable_training_phases = False

    work_dir = "/tmp/test_opponent_integration/"
    os.makedirs(work_dir, exist_ok=True)

    print("=" * 60)
    print(f"track={cfg.AssettoCorsa.track}  car={cfg.AssettoCorsa.car}")
    print(f"enable_opponent={cfg.AssettoCorsa.enable_opponent}")
    print(f"enable_opponent_in_obs={cfg.AssettoCorsa.enable_opponent_in_obs}")
    print("=" * 60)

    env = make_ac_env(cfg=cfg, work_dir=work_dir)
    print(f"\n[env] observation_space.shape = {env.observation_space.shape}")
    print(f"[env] state_dim = {env.state_dim}")

    obs = env.reset()
    print(f"[reset] obs.shape = {obs.shape}")

    opp_dists = []
    margins = []
    track_margins = []
    for step in range(N_STEPS):
        # 简单策略：直行 + 轻微油门，让车慢慢走
        action = np.array([0.0, 0.2, 0.0])  # steer, throttle, brake
        obs, reward, done, info = env.step(action)
        s = env.state
        opp_dists.append(s.get('opp_signed_dist', None))
        margins.append(s.get('margin', None))
        # 单独从 dist_to_border 算 track_margin 验证（不依赖 step 内部）
        tb = float(s.get('dist_to_border', 0.))
        if s.get('out_of_track', False):
            tb = -tb
        track_margins.append(tb)

        if step % 10 == 0:
            print(f"\nstep={step:3d}  reward={reward:+.3f}  done={int(done)}")
            print(f"  ego_pos = ({s.get('world_position_x', 0):.2f}, {s.get('world_position_y', 0):.2f})")
            print(f"  track_margin (dist_to_border, signed) = {tb:+.2f}")
            print(f"  opp_signed_dist = {s.get('opp_signed_dist', None):+.2f}")
            print(f"  margin = min(track, opp) = {s.get('margin', None):+.2f}")
            print(f"  obs[-1] (归一化 opp) = {obs[-1]:+.4f}")
        if done:
            print(f"\n[end] episode terminated at step {step}")
            break

    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    valid_opp = [d for d in opp_dists if d is not None]
    print(f"opp_signed_dist:  min={min(valid_opp):.2f}  max={max(valid_opp):.2f}  mean={np.mean(valid_opp):.2f}")
    print(f"track_margin:     min={min(track_margins):.2f}  max={max(track_margins):.2f}  mean={np.mean(track_margins):.2f}")
    print(f"final margin:     min={min(margins):.2f}  max={max(margins):.2f}  mean={np.mean(margins):.2f}")

    # 验证不变量
    print("\n不变量检查:")
    ok = True
    for i, (m, t, o) in enumerate(zip(margins, track_margins, opp_dists)):
        expected = min(t, o)
        if abs(m - expected) > 1e-3:
            print(f"  ❌ step {i}: margin={m:.3f} != min(track={t:.3f}, opp={o:.3f})={expected:.3f}")
            ok = False
            if i > 3:
                break
    if ok:
        print(f"  ✅ 所有 {len(margins)} 步满足 margin == min(track, opp)")
    expected_dim = env.state_dim
    if obs.shape[0] == expected_dim:
        print(f"  ✅ obs.shape ({obs.shape[0]}) == state_dim ({expected_dim})")
    else:
        print(f"  ❌ obs.shape ({obs.shape[0]}) != state_dim ({expected_dim})")

    env.close()
    print("\n[done]")


if __name__ == "__main__":
    main()
