import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque
import os
from .replay_buffers import ReplayBuffer, Pri_ReplayBuffer


class QNetwork(nn.Module):
    """DQN网络结构"""
    
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
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class DQNAgent:
    """
    DQN智能体，支持优先经验回放和n-step学习
    """
    
    def __init__(self, state_dim, action_dim, replay_buffer_type="uniform", n_step=4, num_envs=1):
        """
        初始化DQN智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            replay_buffer_type: 经验池类型 ("uniform" 或 "prioritized")
            n_step: n-step学习的步数
            num_envs: 并行环境数量
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.qnet = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.best_net = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=2e-4)
        self.batch_size = 64
        
        # 根据类型选择经验池
        if replay_buffer_type == "uniform":
            self.replay_buffer = ReplayBuffer(300000, self.batch_size)
        elif replay_buffer_type == "prioritized":
            self.replay_buffer = Pri_ReplayBuffer(300000, self.batch_size)
        else:
            raise ValueError(f"Unknown replay buffer type: {replay_buffer_type}")
        self.replay_buffer_type = replay_buffer_type
        
        # 超参数
        self.epsilon = 0.1
        self.gamma = 0.99
        self.target_update_freq = 1000
        self.step_count = 0
        self.best_avg_reward = 0
        self.eval_episodes = 3
        self.action_dim = action_dim
        self.update_count = 0
        self.beta_start = 0.4
        self.beta_frames = 100000
        self.beta = self.beta_start
        
        # n-step相关
        self.n_step = n_step
        self.num_envs = num_envs
        self.n_step_buffer = [[] for _ in range(num_envs)]
        
    def choose_action(self, state):
        """选择动作（epsilon-greedy策略）"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.qnet(state_tensor)
                return q_values.argmax().item()
    
    def store_nstep_experience(self, env_idx, state, action, reward, next_state, done):
        """存储n-step经验"""
        self.n_step_buffer[env_idx].append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer[env_idx]) >= self.n_step or done:
            # 计算n-step reward
            n_step_reward = 0
            for i, (_, _, r, _, _) in enumerate(self.n_step_buffer[env_idx]):
                n_step_reward += (self.gamma ** i) * r
            
            # 获取n-step后的状态
            if done:
                n_step_next_state = next_state
            else:
                n_step_next_state = self.n_step_buffer[env_idx][-1][3]
            
            # 存储n-step经验
            self.store_experience(
                self.n_step_buffer[env_idx][0][0],  # 初始状态
                self.n_step_buffer[env_idx][0][1],  # 初始动作
                n_step_reward,
                n_step_next_state,
                done,
                env_idx
            )
            
            # 清空缓冲区
            self.n_step_buffer[env_idx] = []
    
    def store_experience(self, state, action, reward, next_state, done, env_idx=0):
        """存储经验到经验池"""
        if self.replay_buffer_type == "prioritized":
            self.replay_buffer.store(state, action, reward, next_state, done, self.n_step)
        else:
            self.replay_buffer.store(state, action, reward, next_state, done, self.n_step)
    
    def train(self):
        """训练智能体"""
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # 更新beta参数
        self.beta = min(1.0, self.beta_start + self.step_count * (1.0 - self.beta_start) / self.beta_frames)
        
        if self.replay_buffer_type == "prioritized":
            idxs, states, actions, rewards, next_states, dones, n_steps, weights = self.replay_buffer.sample()
        else:
            states, actions, rewards, next_states, dones, n_steps = self.replay_buffer.sample()
            weights = torch.ones(self.batch_size).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.qnet(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma ** n_steps) * next_q_values * (1 - dones)
        
        # 计算损失
        td_errors = torch.abs(current_q_values.squeeze() - target_q_values)
        loss = (weights * F.smooth_l1_loss(current_q_values.squeeze(), target_q_values, reduction='none')).mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnet.parameters(), 10)
        self.optimizer.step()
        
        # 更新目标网络
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.qnet.state_dict())
        
        # 更新优先经验回放的优先级
        if self.replay_buffer_type == "prioritized":
            self.replay_buffer.update(idxs, td_errors.detach().cpu().numpy())
        
        self.update_count += 1
        return self.update_count
    
    def eval(self, env):
        """评估智能体"""
        total_reward = 0
        for _ in range(self.eval_episodes):
            state, _ = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.choose_action(state[0])
                next_state, reward, terminated, truncated, info = env.step([action])
                episode_reward += reward[0]
                state = next_state
                done = terminated[0] or truncated[0]
            total_reward += episode_reward
        return total_reward / self.eval_episodes
    
    def save_model(self, path="./output"):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        torch.save(self.best_net.state_dict(), os.path.join(path, "best_model.pth"))
        print(f"Model saved to {path}")
    
    def reset(self):
        """重置智能体状态"""
        self.step_count = 0
        self.update_count = 0
        self.beta = self.beta_start
        self.n_step_buffer = [[] for _ in range(self.num_envs)] 