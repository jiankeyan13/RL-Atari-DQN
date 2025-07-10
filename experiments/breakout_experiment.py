import random
import numpy as np
import torch
import swanlab
from gymnasium.vector import SyncVectorEnv

from agents.random_agent import RandomAgent
from agents.dqn_agent import DQNAgent
from environments.atari_env import make_env
from utils.config import SEED, DEFAULT_CONFIG, GAME_CONFIGS
from utils.training_utils import train_dqn, test_and_record_video


def run_breakout_experiment():
    """运行 Breakout 游戏的完整实验"""
    # 设置随机种子
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # 游戏配置
    GAME_NAME = "BreakoutNoFrameskip-v4"
    game_config = GAME_CONFIGS[GAME_NAME]

    # 创建智能体（动作维度将在训练函数中自动设置）
    agent = None
    
    # 训练智能体
    agent = train_dqn(GAME_NAME, agent, DEFAULT_CONFIG, game_config)
    
    # 测试并录制视频
    test_and_record_video(agent, GAME_NAME, game_config)


def run_random_policy(game_name, config, game_config):
    """运行随机策略基线"""
    from gymnasium.vector import SyncVectorEnv
    import swanlab
    
    envs = SyncVectorEnv([make_env(game_name) for _ in range(config["num_envs"])])
    action_dim = envs.single_action_space.n

    swanlab.init(
        project="RL-Task",
        experiment_name=game_config["experiment_name"] + "-Random",
        config={
            "action_dim": action_dim,
            "total_frames": config["total_frames"],
            "num_envs": config["num_envs"],
        },
        description="Random Policy Baseline"
    )

    agent = RandomAgent(action_dim)
    frame_count = 0
    episode = 0
    episode_rewards = np.zeros(config["num_envs"])
    recent_rewards = []

    states, _ = envs.reset(seed=42)
    while frame_count < config["total_frames"]:
        actions = np.array([agent.choose_action(s) for s in states])
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions)
        episode_rewards += rewards
        frame_count += config["num_envs"]

        for i in range(config["num_envs"]):
            if terminateds[i] or truncateds[i]:
                recent_rewards.append(episode_rewards[i])
                episode += 1
                episode_rewards[i] = 0
        states = next_states

        if len(recent_rewards) >= 10 and frame_count % 1000 == 0:
            avg_train_reward = np.mean(recent_rewards[-10:])
            print(f"[Random] Frame: {frame_count}, Avg Reward: {avg_train_reward}, Episode: {episode}")
            swanlab.log({
                "train/avg_reward": avg_train_reward,
                "train/episode": episode
            }, step=frame_count)

        if frame_count % config["eval_interval"] == 0 or frame_count >= config["total_frames"]:
            eval_env = SyncVectorEnv([make_env(game_name) for _ in range(1)])
            eval_reward = 0
            for _ in range(3):
                state, _ = eval_env.reset()
                done = False
                total = 0
                while not done:
                    action = agent.choose_action(state[0])
                    next_state, reward, terminated, truncated, info = eval_env.step([action])
                    total += reward[0]
                    state = next_state
                    done = terminated[0] or truncated[0]
                eval_reward += total
            eval_reward /= 3
            eval_env.close()
            swanlab.log({
                "eval/score": eval_reward
            }, step=frame_count)
            print(f"[Random][Eval] Frame: {frame_count}, Eval Reward: {eval_reward}")

    envs.close()


if __name__ == "__main__":
    # 运行 Breakout 实验
    run_breakout_experiment()
    
    # 运行随机策略基线
    GAME_NAME = "BreakoutNoFrameskip-v4"
    game_config = GAME_CONFIGS[GAME_NAME]
    run_random_policy(GAME_NAME, DEFAULT_CONFIG, game_config) 