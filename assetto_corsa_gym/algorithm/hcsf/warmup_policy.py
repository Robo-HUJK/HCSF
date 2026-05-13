"""
预热策略奖励函数 (论文 Appendix C-2)

名义策略 reward (Eq. 17):
    r_nom = (v / c2) * (1 - c1 * d_gap)
    v = ego speed, d_gap = ℓ2 distance to reference path
    c1 = 1/12, c2 = 300

超车策略 reward (Eq. 18-20):
    rover,1 = I{doppo < d_over_oppo} * c3 * (ΔNSP_{t-1} - ΔNSP_t)
    rover,2 = I{rover,1 > 0} * (c1/c2) * v * d_gap
    rover   = r_nom + max(rover,1, 0) + rover,2

超车策略需要对手信息 → 当前代码无对手支持, 标记 TODO.
"""


def compute_nominal_reward(speed, gap, c1=1.0/12.0, c2=300.0):
    """
    名义策略奖励 (论文 Eq. 17).
    鼓励高速度 + 低偏离参考线.

    Args:
        speed: ego vehicle speed (m/s)
        gap:   distance to reference path (m)
    Returns:
        scalar reward
    """
    return (speed / c2) * (1.0 - c1 * abs(gap))


def compute_overtaking_reward(speed, gap, doppo, d_over_oppo,
                               delta_nsp, c1=1.0/12.0, c2=300.0,
                               c3=600.0):
    """
    超车策略奖励 (论文 Eq. 18-20). 暂未使用 — 需要对手支持.

    Args:
        speed:        ego speed
        gap:          distance to reference path
        doppo:        distance to nearest opponent
        d_over_oppo:  threshold for overtaking reward (100m)
        delta_nsp:    change in normalized spline position difference
                      (= ΔNSP_{t-1} - ΔNSP_t, positive when closing in)
    Returns:
        scalar reward
    """
    r_nom = compute_nominal_reward(speed, gap, c1, c2)

    # Eq. 18: overtaking reward when close to opponent
    if doppo < d_over_oppo:
        rover_1 = c3 * delta_nsp
    else:
        rover_1 = 0.0

    # Eq. 19: relax path-following penalty when overtaking
    if rover_1 > 0:
        rover_2 = (c1 / c2) * speed * abs(gap)
    else:
        rover_2 = 0.0

    # Eq. 20: total overtaking reward
    rover = r_nom + max(rover_1, 0) + rover_2
    return rover


# 论文 Table V 超参数
WARMUP_CONFIG = {
    'T_warmup_max': 25.0,       # max warmup duration (s)
    'P_over': 0.6,              # probability of overtaking policy in warmup
    'P_oppo': 0.25,             # probability of early termination near opponent
    'doppo_min': 6.0,           # min opponent distance for termination (m)
    'doppo_max': 36.0,          # max opponent distance for termination (m)
    'P_brake_epi': 0.4,         # probability of enabling brake termination per episode
    'P_brake_step': 0.25,       # probability of brake termination per step
    'u_brake': 0.6,             # threshold for heavy braking input
    'v_warmup': 40.0,           # speed threshold for brake termination (m/s)
    'c1': 1.0 / 12.0,
    'c2': 300.0,
    'd_over_oppo': 100.0,       # overtaking distance threshold (m)
    'c3': 600.0,
}

INIT_CONFIG = {
    'P_term': 0.2,              # probability of ending init when Q < Q_term
    'Q_init_term': 2.0,         # Q threshold for dangerous states
    'P_FT': 0.4,                # probability of full-throttle mode
    'P_adv': 0.3,               # probability of adversarial init
    'P_rand': 0.3,              # probability of random init
    'P_mix': 0.4,               # probability of mixed init
}
