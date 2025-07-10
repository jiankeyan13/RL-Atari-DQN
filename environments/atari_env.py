import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo
import numpy as np


class FireResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Atari游戏的火力重置包装器
    某些Atari游戏需要先按FIRE键才能开始游戏
    """

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
    """
    创建训练用的Atari环境
    
    Args:
        game_name: 游戏名称
        
    Returns:
        环境创建函数
    """
    def _thunk():
        env = gym.make(game_name, frameskip=1)
        env = FireResetEnv(env)
        env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, noop_max=10, terminal_on_life_loss=True)
        env = FrameStackObservation(env, stack_size=4)
        return env
    return _thunk


def make_env_with_render(game_name, video_dir):
    """
    创建支持渲染和录制的Atari环境
    
    Args:
        game_name: 游戏名称
        video_dir: 视频保存目录
        
    Returns:
        配置好的环境
    """
    env = gym.make(game_name, render_mode='rgb_array', frameskip=1)
    env = FireResetEnv(env)
    env = AtariPreprocessing(env, grayscale_obs=True, scale_obs=True, terminal_on_life_loss=True)
    env = FrameStackObservation(env, 4)
    env = RecordVideo(env, video_dir, episode_trigger=lambda x: True)
    return env 