import os

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2"

# 随机种子
SEED = 3407

# 训练超参数
DEFAULT_CONFIG = {
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.9999,
    "total_frames": 5_000_000,
    "batch_size": 256,
    "replay_buffer_size": 300000,
    "learning_start": 1500,
    "eval_interval": 25000,
    "target_update_freq": 625,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "num_envs": 16,
    "state_dim": 4,
    "eval_episodes": 3,
}

# 游戏特定配置
GAME_CONFIGS = {
    "BreakoutNoFrameskip-v4": {
        "experiment_name": "Breakout",
        "model_path": "./output/Breakout"
    },
    "PongNoFrameskip-v4": {
        "experiment_name": "DQN-Pong", 
        "model_path": "./output/Pong"
    }
} 