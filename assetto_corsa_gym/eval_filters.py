"""
Phase 3 滤波器验证 — 高噪声模式, 强制触发干预.
"""
import sys, os
sys.path.extend([os.path.abspath('./assetto_corsa_gym'), './algorithm/discor', './algorithm'])

import torch, numpy as np
from omegaconf import OmegaConf
from discor.network import TwinnedStateActionFunction, GaussianPolicy
from hcsf.lrsf_filter import LRSF
from hcsf.hcsf_filter import HCSF
import AssettoCorsaEnv.assettoCorsa as assettoCorsa

device = 'cuda'
state_dim, action_dim = 125, 3

model_dir = '/home/wyb/car/AssettoCorsaGymDataSet/data_sets/ks_barcelona-layout_gp/bmw_z4_gt3/20240404_SAC_10M/model/final'
q_net = TwinnedStateActionFunction(state_dim, action_dim, [256,256,256]).to(device).eval()
q_net.load(f'{model_dir}/online_q_net.pth')
policy_net = GaussianPolicy(state_dim, action_dim, [256,256,256]).to(device).eval()
policy_net.load(f'{model_dir}/policy_net.pth')

config = OmegaConf.load('config.yml')
env = assettoCorsa.make_ac_env(cfg=config, work_dir='/tmp/ac_hcsf_test')

hcsf = HCSF(q_net, policy_net, device, v_net=None)
lrsf = LRSF(q_net, policy_net, device, v_net=None)

print('=== HCSF vs LRSF: high-noise human ===')

for fname, fobj in [('HCSF', hcsf), ('LRSF', lrsf)]:
    state = env.reset()
    records = []
    for s in range(300):
        state_t = torch.tensor(state[None], dtype=torch.float, device=device)
        with torch.no_grad():
            _, _, det = policy_net(state_t)
        u_h = det.cpu().numpy()[0] + np.random.normal(0, 0.8, 3)  # 高噪声
        u_h = np.clip(u_h, -1, 1)

        u_f, info = fobj.filter(state, u_h)
        env.set_actions(u_f)
        ns, _, done, _ = env.step(action=None)

        records.append({'im': info.get('im', 0.0), 'v': info['v_value'],
                        'int': info['intervened'], 'sat': info.get('n_satisfied', 0)})
        state = ns
        if done:
            break

    intv = sum(1 for r in records if r['int'])
    avg_im = np.mean([r['im'] for r in records])
    avg_v = np.mean([r['v'] for r in records])
    avg_sat = np.mean([r['sat'] for r in records])
    print(f'  [{fname}] steps={len(records)}  V_avg={avg_v:.1f}  '
          f'IM_avg={avg_im:.4f}  sat_avg={avg_sat:.0f}/2000  '
          f'intervened={intv}/{len(records)} ({100*intv/max(1,len(records)):.0f}%)')

env.close()
print('Done')
