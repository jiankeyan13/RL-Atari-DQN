import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
import os
import numpy as np
import swanlab
from gymnasium.vector import SyncVectorEnv
import torch
import ale_py  # 添加这个导入
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"  # type: ignore[attr-defined]
        assert len(env.unwrapped.get_action_meanings()) >= 3  # type: ignore[attr-defined]

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(1)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, _ = self.env.step(2)
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, {}

def make_env(game_name):
    def _thunk():
        env = gym.make(game_name, frameskip=1)
        env = FireResetEnv(env)
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True,noop_max=10, terminal_on_life_loss=True)
        env = FrameStackObservation(env, stack_size=4)
        return env
    return _thunk

def make_env_with_render(game_name, video_dir):
    env = gym.make(game_name, render_mode='rgb_array', frameskip=1)
    env = FireResetEnv(env)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
    env = FrameStackObservation(env, 4)
    env = RecordVideo(env, video_dir, episode_trigger=lambda x: True)
    return env
    

def train_dqn(game_name, agent, config, game_config):
    """
    训练DQN智能体的完整流程
    
    Args:
        game_name: 游戏名称
        agent: DQN智能体
        config: 训练配置
        game_config: 游戏特定配置
    """
    # 创建多环境（thunk风格）
    envs = SyncVectorEnv([make_env(game_name) for _ in range(config["num_envs"])])
    state_dim = config["state_dim"]
    action_dim = envs.single_action_space.n
    
    print(f"state_dim: {state_dim}, action_dim: {action_dim}")

    # 重新创建智能体以匹配正确的动作维度
    from common.models import DQNAgent
    agent = DQNAgent(state_dim, action_dim, replay_buffer_type="prioritized", n_step=4, num_envs=config["num_envs"])
    print(agent.replay_buffer_type)
    # 配置智能体
    agent.batch_size = config["batch_size"]
    agent.replay_buffer = agent.replay_buffer.__class__(config["replay_buffer_size"], agent.batch_size)
    agent.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent.qnet = agent.qnet.to(agent.device)
    agent.target_net = agent.target_net.to(agent.device)
    agent.best_net = agent.best_net.to(agent.device)
    agent.epsilon = config["epsilon_start"]
    agent.target_update_freq = config["target_update_freq"]
    # 初始化swanlab
    swanlab.init(
        project="RL-Task",
        experiment_name=game_config["experiment_name"],
        config={
            "state_dim": state_dim,
            "action_dim": action_dim,
            "batch_size": agent.batch_size,
            "gamma": agent.gamma,
            "epsilon_start": config["epsilon_start"],
            "epsilon_end": config["epsilon_end"],
            "epsilon_decay": config["epsilon_decay"],
            "target_update_freq": agent.target_update_freq,
            "replay_buffer_size": config["replay_buffer_size"],
            "learning_rate": agent.optimizer.param_groups[0]['lr'],
            "total_frames": config["total_frames"],
            "num_envs": config["num_envs"],
            "eval_interval": config["eval_interval"],
        },
        description=game_config.get("description", "DQN Training")
    )


    # 预填充经验池
    states, _ = envs.reset(seed=42)

    for i in range(config["learning_start"]):
        actions = np.random.randint(0, action_dim, size=config["num_envs"])
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        for i in range(config["num_envs"]):
            done = terminateds[i] or truncateds[i]
            agent.store_experience(states[i], actions[i], rewards[i], next_states[i], done, env_idx=i)
        states = next_states
    print("Fill with 16000")

    # 训练循环
    frame_count = 0
    episode = 0
    last_eval_frame = 0
    episode_rewards = np.zeros(config["num_envs"])
    recent_rewards = []
    all_rewards = []

    states, _ = envs.reset(seed=42)
    while frame_count < config["total_frames"]:
        # 动态调整目标网络同步频率
        if frame_count > 1_000_000:
            agent.target_update_freq = 5000
        elif frame_count > 500_000:
            agent.target_update_freq = 2000
        else:
            agent.target_update_freq = 1000
        actions = []
        for i in range(config["num_envs"]):
            actions.append(agent.choose_action(states[i]))
        actions = np.array(actions)
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        for i in range(config["num_envs"]):
            done = terminateds[i] or truncateds[i]
            agent.store_experience(states[i], actions[i], rewards[i], next_states[i], done, env_idx=i)
        updates = agent.train()
        episode_rewards += rewards
        frame_count += config["num_envs"]
        # epsilon指数衰减
        agent.epsilon = max(config["epsilon_end"], agent.epsilon * config["epsilon_decay"])
        # 统计episode结束
        for i in range(config["num_envs"]):
            if terminateds[i] or truncateds[i]:
                recent_rewards.append(episode_rewards[i])
                episode += 1
                episode_rewards[i] = 0
        states = next_states
        # 训练日志：每1000帧上传一次
        if len(recent_rewards) >= 10 and frame_count % 1000 == 0:
            avg_train_reward = np.mean(recent_rewards[-10:])
            print(f"Frame: {frame_count}, Avg Reward: {avg_train_reward}, Episode: {episode}")
            swanlab.log({
                "train/avg_reward": avg_train_reward,
                "train/update": updates, 
                "train/epsilon": agent.epsilon,
                "train/replay_buffer_size": len(agent.replay_buffer),
                "train/episode": episode
            }, step=frame_count)

        # 定期评估
        if frame_count - last_eval_frame >= config["eval_interval"] or frame_count >= config["total_frames"]:
            print("进入评估模式")
            eval_env = SyncVectorEnv([make_env(game_name) for _ in range(1)])
            eval_reward = agent.eval(eval_env)
            eval_env.close()
            swanlab.log({
                "eval/score": eval_reward
            }, step=frame_count)
            print(f"[Eval] Frame: {frame_count}, Eval Reward: {eval_reward}")
            last_eval_frame = frame_count
            if eval_reward > agent.best_avg_reward:
                agent.best_avg_reward = eval_reward
                agent.best_net.load_state_dict({k: v.clone() for k, v in agent.qnet.state_dict().items()})
                agent.save_model(path=game_config["model_path"])
                print(f"New best model saved with average eval reward: {eval_reward}")

    envs.close()
    return agent

def test_and_record_video(agent, game_name, game_config, video_dir="./dqn_videos", num_episodes=3, max_steps=1500):
    """
    测试模型并录制视频
    
    Args:
        agent: 训练好的DQN智能体
        game_name: 游戏名称
        make_env_func: 环境创建函数
        video_dir: 视频保存目录
        num_episodes: 测试的episode数量
        max_steps: 每个episode的最大步数
    """
    # 关闭探索策略
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    # 加载最佳模型
    best_model_path = os.path.join(game_config["model_path"], "best_model.pth")
    if os.path.exists(best_model_path):
        agent.qnet.load_state_dict(torch.load(best_model_path, map_location=agent.device))
        print("Loaded best model from disk for video recording.")
    else:
        print("Best model file not found, using current qnet parameters.")
    
    # 创建测试环境（使用支持渲染的环境创建函数）
    test_env = make_env_with_render(game_name, video_dir)
    # test_env = gym.wrappers.RecordVideo(test_env, video_dir, episode_trigger=lambda x: True)
    
    # 测试并录制
    for episode in range(num_episodes):
        state = test_env.reset()[0]
        total_reward = 0
        steps = 0
        while True:
            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state
            steps += 1
            if done or steps >= max_steps:
                break
        print(f"Test Episode: {episode}, Reward: {total_reward}")
    
    test_env.close()
    print(f"Videos saved to {video_dir}")
    
    # 恢复原来的epsilon
    agent.epsilon = original_epsilon 