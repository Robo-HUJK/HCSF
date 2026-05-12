from collections import deque
import numpy as np
import torch

import logging
logger = logging.getLogger(__name__)


class NStepBuffer:
    def __init__(self, gamma=0.99, nstep=3):
        assert isinstance(gamma, float) and 0 < gamma < 1.0
        assert isinstance(nstep, int) and nstep > 0

        self._discounts = [gamma ** i for i in range(nstep)]
        self._nstep = nstep
        self.reset()

    def append(self, state, action, reward, margin=None):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        if margin is not None:
            self._margins.append(margin)

    def get(self):
        assert len(self._rewards) > 0

        state = self._states.popleft()
        action = self._actions.popleft()
        reward = self._nstep_reward()
        margin = self._margins.popleft() if len(self._margins) > 0 else None
        return state, action, reward, margin

    def _nstep_reward(self):
        reward = np.sum([
            r * d for r, d in zip(self._rewards, self._discounts)])
        self._rewards.popleft()
        return reward

    def reset(self):
        self._states = deque(maxlen=self._nstep)
        self._actions = deque(maxlen=self._nstep)
        self._rewards = deque(maxlen=self._nstep)
        self._margins = deque(maxlen=self._nstep)

    def is_empty(self):
        return len(self._rewards) == 0

    def is_full(self):
        return len(self._rewards) == self._nstep

    def __len__(self):
        return len(self._rewards)


class ReplayBuffer:
    def __init__(self, memory_size, state_shape, action_shape, gamma=0.99, nstep=1):
        assert isinstance(memory_size, int) and memory_size > 0
        assert isinstance(state_shape, tuple)
        assert isinstance(action_shape, tuple)
        assert isinstance(gamma, float) and 0 < gamma < 1.0
        assert isinstance(nstep, int) and nstep > 0

        self._memory_size = memory_size
        self._state_shape = state_shape
        self._action_shape = action_shape
        self._gamma = gamma
        self._nstep = nstep
        self._reset()

    def _reset(self):
        self._n = 0
        self._p = 0

        self._states = np.empty(
            (self._memory_size, ) + self._state_shape, dtype=np.float32)
        self._next_states = np.empty(
            (self._memory_size, ) + self._state_shape, dtype=np.float32)
        self._actions = np.empty(
            (self._memory_size, ) + self._action_shape, dtype=np.float32)

        self._rewards = np.empty((self._memory_size, 1), dtype=np.float32)
        self._dones = np.empty((self._memory_size, 1), dtype=np.float32)
        self._margins = np.empty((self._memory_size, 1), dtype=np.float32)

        if self._nstep != 1:
            self._nstep_buffer = NStepBuffer(self._gamma, self._nstep)
        logger.info(f"Replay buffer initialized for {self._memory_size} samples")

    def append(self, state, action, reward, next_state, terminated, episode_done=None, margin=None):
        """
        done (masked_done): False if the agent reach time horizons. Else = done
        margin: g(state) — 当前状态的安全裕度 (论文 Algorithm 1 第 9 行)
        """
        if self._nstep != 1:
            self._nstep_buffer.append(state, action, reward, margin)

            if self._nstep_buffer.is_full():
                state, action, reward, margin = self._nstep_buffer.get()
                self._append(state, action, reward, next_state, terminated, margin)

            if terminated or episode_done:
                while not self._nstep_buffer.is_empty():
                    state, action, reward, margin = self._nstep_buffer.get()
                    self._append(state, action, reward, next_state, terminated, margin)

        else:
            self._append(state, action, reward, next_state, terminated, margin)

    def _append(self, state, action, reward, next_state, done, margin=None):
        self._states[self._p, ...] = state
        self._actions[self._p, ...] = action
        self._rewards[self._p, ...] = reward
        self._next_states[self._p, ...] = next_state
        self._dones[self._p, ...] = done

        # 默认安全裕度: 正值 (车初始状态在赛道内)
        self._margins[self._p, ...] = margin if margin is not None else 1.0

        self._n = min(self._n + 1, self._memory_size)
        self._p = (self._p + 1) % self._memory_size

    def sample(self, batch_size, device=torch.device('cpu')):
        assert isinstance(batch_size, int) and batch_size > 0

        idxes = self._sample_idxes(batch_size)
        return self._sample_batch(idxes, batch_size, device)

    def _sample_idxes(self, batch_size):
        return np.random.randint(low=0, high=self._n, size=batch_size)

    def _sample_batch(self, idxes, batch_size, device):
        states = torch.tensor(
            self._states[idxes], dtype=torch.float, device=device)
        actions = torch.tensor(
            self._actions[idxes], dtype=torch.float, device=device)
        rewards = torch.tensor(
            self._rewards[idxes], dtype=torch.float, device=device)
        dones = torch.tensor(
            self._dones[idxes], dtype=torch.float, device=device)
        next_states = torch.tensor(
            self._next_states[idxes], dtype=torch.float, device=device)
        margins = torch.tensor(
            self._margins[idxes], dtype=torch.float, device=device)

        return states, actions, rewards, next_states, dones, margins

    def __len__(self):
        return self._n


class EnsembleBuffer(ReplayBuffer):
    """
    Ensemble of an offline dataloader and an online replay buffer.
    """
    def __init__(self, memory_size, state_shape, action_shape, offline_buffer_size, gamma=0.99, nstep=1):
        # Initialize the offline buffer with the specified offline_buffer_size
        self._offline = ReplayBuffer(offline_buffer_size, state_shape, action_shape, gamma, nstep)

        # Initialize the online buffer using the parent class constructor
        super().__init__(memory_size, state_shape, action_shape, gamma, nstep)
        self._online = False

    def online(self, enable):
        """Enable or disable sampling from the online buffer."""
        self._online = enable
        if enable:
            logger.info("Switching to Online buffer.")

    def append(self, state, action, reward, next_state, terminated, episode_done=None):
        if self._online:
            super().append(state, action, reward, next_state, terminated, episode_done)
        else:
            self._offline.append(state, action, reward, next_state, terminated, episode_done)

    def __len__(self):
        offline_len = len(self._offline)
        online_len = super().__len__()
        logger.info(f"Offline buffer size: {offline_len}, Online buffer size: {online_len}.")
        return offline_len + online_len

    def sample(self, batch_size, device=torch.device('cpu')):
        """Sample a batch of data from the two buffers."""
        obs0, action0, reward0, next_states0, dones0, margins0 = self._offline.sample(batch_size // 2, device)
        obs1, action1, reward1, next_states1, dones1, margins1 = (super() if self._online else self._offline).sample(batch_size // 2, device)

        # Concatenate samples from both buffers
        states = torch.cat([obs0, obs1], dim=0)
        actions = torch.cat([action0, action1], dim=0)
        rewards = torch.cat([reward0, reward1], dim=0)
        next_states = torch.cat([next_states0, next_states1], dim=0)
        dones = torch.cat([dones0, dones1], dim=0)
        margins = torch.cat([margins0, margins1], dim=0)

        return states, actions, rewards, next_states, dones, margins
