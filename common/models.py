import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os

class SumTree:
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

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(state_dim, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        if x.dim() == 3: 
            x = x.unsqueeze(0)
        # 假设输入已是 [N, C, H, W] 且已归一化，无需 permute
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

class ReplayBuffer:
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
    def __init__(self, max_size=100000, batch_size=128, alpha=0.9, epsilon=0.01, beta=0.4):
        self.tree = SumTree(max_size)
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.alpha = alpha  # priority exponent
        self.epsilon = epsilon  # small constant to avoid zero priority
        self.prio_max = 1.5  # max priority for new experiences - 更合理的初始值
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
        # 对于Atari游戏，使用更保守的更新策略
        self.prio_max = max(self.prio_max * 0.995 + current_max_error * 0.005, current_max_error)
        
        for i, idx in enumerate(idxs):
            p = (np.abs(errors[i]) + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
    
    def __len__(self):
        return self.tree.n_entries

class DQNAgent:
    def __init__(self, state_dim, action_dim, replay_buffer_type="uniform", n_step=4, num_envs=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.qnet = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.qnet.state_dict()) # 初始化target_net
        self.best_net = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(),lr=2e-4)
        self.batch_size = 64
        
        # 根据类型选择经验池
        if replay_buffer_type == "uniform":
            self.replay_buffer = ReplayBuffer(300000, self.batch_size)
        elif replay_buffer_type == "prioritized":
            self.replay_buffer = Pri_ReplayBuffer(300000, self.batch_size)
        else:
            raise ValueError(f"Unknown replay buffer type: {replay_buffer_type}")
        self.replay_buffer_type = replay_buffer_type
        #超参数
        self.epsilon = 0.1
        self.gamma = 0.99
        self.target_update_freq = 1000  # 更新target_net的频率
        self.step_count = 0
        self.best_avg_reward = 0
        self.eval_episodes = 3 # 评估次数
        self.action_dim = action_dim
        self.update_count = 0
        self.beta_start = 0.4
        self.beta_frames = 100000  # 退火到1.0所需帧数
        self.beta = self.beta_start
        # n-step相关
        self.n_step = n_step
        self.num_envs = num_envs
        self.nstep_queues = [deque() for _ in range(num_envs)]  # 不设maxlen，手动出队

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            # state: [4, 84, 84]
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # [1, 4, 84, 84]
            q_values = self.qnet(state_tensor)
            return q_values.cpu().detach().numpy().argmax()
    
    def store_nstep_experience(self, env_idx, state, action, reward, next_state, done):
        q = self.nstep_queues[env_idx]
        q.append((state, action, reward, next_state, done))
        if done:
            # 补齐所有剩余片段
            while q:
                nstep_list = list(q)
                s_t, a_t = nstep_list[0][0], nstep_list[0][1]
                n_step_reward = 0
                discount = 1
                for i, (_, _, r, _, d) in enumerate(nstep_list):
                    n_step_reward += discount * r
                    discount *= self.gamma
                    if d:
                        break
                next_n_state = nstep_list[-1][3]
                n_done = True
                n = len(nstep_list)
                self.replay_buffer.store(s_t, a_t, n_step_reward, next_n_state, n_done, n)
                q.popleft()
        elif len(q) >= self.n_step:
            nstep_list = list(q)[:self.n_step]
            s_t, a_t = nstep_list[0][0], nstep_list[0][1]
            n_step_reward = 0
            discount = 1
            for i, (_, _, r, _, d) in enumerate(nstep_list):
                n_step_reward += discount * r
                discount *= self.gamma
                if d:
                    break
            if not nstep_list[-1][4]:
                next_n_state = nstep_list[-1][3]
                n_done = False
            else:
                next_n_state = nstep_list[-1][3]
                n_done = True
            n = len(nstep_list)
            self.replay_buffer.store(s_t, a_t, n_step_reward, next_n_state, n_done, n)
            q.popleft()

    def store_experience(self, state, action, reward, next_state, done, env_idx=0):
        # 兼容单环境和多环境
        self.store_nstep_experience(env_idx, state, action, reward, next_state, done)
    
    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        if self.replay_buffer_type == "uniform":
            states, actions, rewards, next_states, dones, n_steps_tensor = self.replay_buffer.sample()
        else:
            self.beta = min(1.0, self.beta + (1.0 - self.beta_start) / self.beta_frames)
            self.replay_buffer.beta = self.beta
            idxs, states, actions, rewards, next_states, dones, n_steps_tensor, weights = self.replay_buffer.sample()
        # n-step TD目标
        if self.replay_buffer_type == "prioritized":
            target_q = rewards + torch.pow(self.gamma, n_steps_tensor.float()) * self.target_net(next_states).max(1)[0] * (1 - dones)
            individual_losses = F.smooth_l1_loss(
                self.qnet(states).gather(1, actions.unsqueeze(1)).squeeze(),
                target_q,
                reduction='none')
            loss = (weights * individual_losses).mean()
        else:
            current_q = self.qnet(states).gather(1, actions.unsqueeze(1)).squeeze()
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target_q = rewards + torch.pow(self.gamma, n_steps_tensor.float()) * next_q * (1 - dones)
            loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), max_norm=1.0)
        self.optimizer.step()
        if self.replay_buffer_type == "prioritized":
            td_errors = torch.abs(
                target_q - self.qnet(states).gather(1, actions.unsqueeze(1)).squeeze()
            ).cpu().detach().numpy()
            self.replay_buffer.update(idxs, td_errors)
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.update_count += 1
            self.target_net.load_state_dict({
                k:v.clone() for k, v in self.qnet.state_dict().items()
            })
        return self.update_count

    def eval(self, env):
        original_epsilon = self.epsilon
        self.epsilon = 0 #关闭探索模式
        total_rewards = []
        
        # 检查是否为向量化环境
        is_vector_env = hasattr(env, 'num_envs')
        
        for _ in range(self.eval_episodes):
            if is_vector_env:
                # 向量化环境处理
                states, _ = env.reset()
                episode_reward = 0
                step_count = 0
                while True:
                    # 为每个环境选择动作
                    actions = []
                    for i in range(env.num_envs):
                        if isinstance(states, np.ndarray) and states.ndim == 4 and states.shape[0] == env.num_envs:
                            state = states[i]  # [num_envs, 4, 84, 84] -> [4, 84, 84]
                        else:
                            state = states[i]
                        actions.append(self.choose_action(state))
                    actions = np.array(actions)
                    
                    next_states, rewards, terminateds, truncateds, infos = env.step(actions)
                    done = terminateds[0] or truncateds[0]  # 假设只有一个环境
                    episode_reward += rewards[0]
                    states = next_states
                    step_count += 1
                    if done or step_count >= 2000:
                        break
            else:
                # 单个环境处理
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
                if isinstance(state, np.ndarray) and state.ndim == 4 and state.shape[0] == 1:
                    state = state[0]  # [1, 4, 84, 84] -> [4, 84, 84]
                episode_reward = 0
                step_count = 0
                while True:
                    action = self.choose_action(state)
                    next_state, reward, terminated, truncated, info = env.step(action)
                    if isinstance(next_state, tuple):
                        next_state = next_state[0]
                    if isinstance(next_state, np.ndarray) and next_state.ndim == 4 and next_state.shape[0] == 1:
                        next_state = next_state[0]
                    done = terminated or truncated
                    episode_reward += reward
                    state = next_state
                    step_count += 1
                    if done or step_count >= 1500:
                        break
            total_rewards.append(episode_reward)
        self.epsilon = original_epsilon
        return np.mean(total_rewards)

    def save_model(self, path = "./output"):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.qnet.state_dict(), os.path.join(path, "best_model.pth"))
        print(f"Model saved to {path}") 