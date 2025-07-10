import torch
import random
import numpy as np
from collections import deque


class SumTree:
    """优先经验回放的SumTree数据结构"""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class ReplayBuffer:
    """标准经验回放缓冲区"""
    
    def __init__(self, max_size=100000, batch_size=128):
        self.buffer = deque(maxlen=max_size)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def store(self, state, action, reward, next_state, done, n):
        self.buffer.append((state, action, reward, next_state, done, n))
    
    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones, n_steps = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor([float(d) for d in dones]).to(self.device)
        n_steps_tensor = torch.LongTensor(n_steps).to(self.device)
        return states, actions, rewards, next_states, dones, n_steps_tensor
    
    def __len__(self):
        return len(self.buffer)


class Pri_ReplayBuffer:
    """优先经验回放缓冲区"""
    
    def __init__(self, max_size=100000, batch_size=128, alpha=0.9, epsilon=0.01, beta=0.4):
        self.tree = SumTree(max_size)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha  # priority exponent
        self.epsilon = epsilon  # small constant to avoid zero priority
        self.prio_max = 1.5  # max priority for new experiences
        self.beta = beta  # IS权重的beta参数
    
    def store(self, state, action, reward, next_state, done, n):
        data = (state, action, reward, next_state, done, n)
        p = (np.abs(self.prio_max) + self.epsilon) ** self.alpha
        self.tree.add(p, data)
    
    def sample(self):
        states, actions, rewards, next_states, dones, n_steps = [], [], [], [], [], []
        idxs = []
        priorities = []
        segment = self.tree.total() / self.batch_size
        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            state, action, reward, next_state, done, n = data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
            n_steps.append(n)
            idxs.append(idx)
            priorities.append(p)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor([float(d) for d in dones]).to(self.device)
        n_steps_tensor = torch.LongTensor(n_steps).to(self.device)
        # 计算IS权重
        probs = np.array(priorities) / self.tree.total()
        N = self.tree.n_entries
        weights = (N * probs) ** (-self.beta)
        weights /= weights.max()  # 归一化
        weights = torch.FloatTensor(weights).to(self.device)
        return idxs, states, actions, rewards, next_states, dones, n_steps_tensor, weights
    
    def update(self, idxs, errors):
        # 动态更新最大优先级，使用滑动平均来平滑变化
        current_max_error = max(np.abs(errors))
        self.prio_max = max(self.prio_max * 0.995 + current_max_error * 0.005, current_max_error)
        
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
    
    def __len__(self):
        return self.tree.n_entries 