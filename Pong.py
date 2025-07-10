from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py

env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=4)

model = DQN("CnnPolicy", env, buffer_size=500000, verbose=1, tensorboard_log="./dqn_pong/")
model.learn(total_timesteps=10_000)

model.save("dqn_pong")