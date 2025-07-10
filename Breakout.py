"""
Breakout 游戏强化学习实验
使用模块化的代码结构运行 DQN 训练和随机基线对比
"""

from experiments.breakout_experiment import run_breakout_experiment, run_random_policy
from utils.config import GAME_CONFIGS, DEFAULT_CONFIG


if __name__ == "__main__":
    # 游戏配置
    GAME_NAME = "BreakoutNoFrameskip-v4"
    game_config = GAME_CONFIGS[GAME_NAME]
    
    # 运行 Breakout 实验（DQN 训练 + 视频录制）
    run_breakout_experiment()
    
    # 运行随机策略基线
    run_random_policy(GAME_NAME, DEFAULT_CONFIG, game_config)