import torch
import torch.nn as nn
import random 
import torch.optim as optim
from collections import deque
import swanlab
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2" 

SEED = 1110
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.qnet = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.qnet.state_dict())
        self.optimizer =  optim.Adam(self.qnet.parameters(), lr=3e-4) #只优化qnet
        self.best_net = QNetwork(state_dim, action_dim).to(self.device)
        self.replay_buffer = deque(maxlen=3000)

        self.batchsize = 64
        self.epsilon = 0.1
        self.gemma = 0.99
        self.step_count = 0
        self.target_update_freq = 100
        self.reward = 0
        self.best_avg_reward = 0
        self.eval_episodes = 5 # 评估次数
    
    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.qnet(state_tensor)
            return q_values.cpu().detach().numpy().argmax()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batchsize: return
        batch = random.sample(self.replay_buffer, self.batchsize)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        current_q = self.qnet(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gemma * next_q * (1-dones)
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count +=1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict({
                k: v.clone() for k, v in self.qnet.state_dict().items()
            })

    def save_model(self, path="./output/best_model.pth"):
        if not os.path.exists("./output"):
            os.makedirs("./output")
        torch.save(self.qnet.state_dict(), path)
        print(f"Model saved to {path}")
    
    def eval(self, env):
        original_epsilon = self.epsilon
        self.epsilon = 0 #关闭探索模式
        total_rewards = []
        for _ in range(self.eval_episodes):
            state = env.reset()[0]
            episode_reward = 0
            while True:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if done or episode_reward >=5e4: 
                    break
            total_rewards.append(episode_reward)
        self.epsilon = original_epsilon
        return np.mean(total_rewards)



# 初始化环境
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQNAgent(state_dim, action_dim)

swanlab.init(
    project="DQN",
    experiment_name="DQN-CartPole-mini",
    config={
        "state_dim": state_dim,
        "action_dim": action_dim,
        "batch_size": agent.batchsize,
        "gamma": agent.gemma,
        "epsilon": agent.epsilon,
        "update_target_freq": agent.target_update_freq,
        "replay_buffer_size": agent.replay_buffer.maxlen,
        "learning_rate": agent.optimizer.param_groups[0]['lr'],
        "episode": 1600,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.9995,
    },
    description="初始化目标网络和当前网络一致，避免网络不一致导致的训练波动"
)

agent.epsilon = swanlab.config["epsilon_start"]
for episode in range(swanlab.config["episode"]):
    state = env.reset()[0]
    total_reward = 0
  
    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.train()

        total_reward += reward
        state = next_state
        if done or total_reward > 5000:
            break
  
    # epsilon是探索系数，随着每一轮训练，epsilon 逐渐减小
    agent.epsilon = max(swanlab.config["epsilon_end"], agent.epsilon * swanlab.config["epsilon_decay"])  
  
    # 每10个episode评估一次模型
    if episode % 10 == 0:
        eval_env = gym.make('CartPole-v1')
        avg_reward = agent.eval(eval_env)
        eval_env.close()
      
        if avg_reward > agent.best_avg_reward:
            agent.best_avg_reward = avg_reward
            # 深拷贝当前最优模型的参数
            agent.best_net.load_state_dict({k: v.clone() for k, v in agent.qnet.state_dict().items()})
            agent.save_model(path=f"./output/best_model.pth")
            print(f"New best model saved with average reward: {avg_reward}")

    print(f"Episode: {episode}, Train Reward: {total_reward}, Best Eval Avg Reward: {agent.best_avg_reward}")
  
    swanlab.log(
        {
            "train/reward": total_reward,
            "eval/best_avg_reward": agent.best_avg_reward,
            "train/epsilon": agent.epsilon
        },
        step=episode,
    )

# 测试并录制视频
agent.epsilon = 0  # 关闭探索策略
test_env = gym.make('CartPole-v1', render_mode='rgb_array')
test_env = RecordVideo(test_env, "./dqn_videos", episode_trigger=lambda x: True)  # 保存所有测试回合
agent.qnet.load_state_dict(agent.best_net.state_dict())  # 使用最佳模型

for episode in range(3):  # 录制3个测试回合
    state = test_env.reset()[0]
    total_reward = 0
    steps = 0
  
    while True:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = test_env.step(action)
        total_reward += reward
        state = next_state
        steps += 1
      
        # 限制每个episode最多1500步,约30秒,防止录制时间过长
        if done or steps >= 1500:
            break
  
    print(f"Test Episode: {episode}, Reward: {total_reward}")

test_env.close()



# if __name__ == '__main__':
#     # --- 验证任务1: 实现 QNetwork 类 ---
#     STATE_DIM = 4
#     ACTION_DIM = 2
    
#     # 实例化网络并传入一个假数据
#     q_network = QNetwork(state_dim=STATE_DIM, action_dim=ACTION_DIM)
#     q_values = q_network(torch.randn(1, STATE_DIM))
    
#     # 打印网络输出的维度，以验证其结构是否正确
#     print(f"Task 1 Verification: Q-Network output shape is {q_values.shape}")
#     print(state_dim)
#     print(action_dim)