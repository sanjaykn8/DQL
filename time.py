import ale_py
import gymnasium as gym
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

# register all ALE environments
gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5")
env = ResizeObservation(env, (84, 84))
env = GrayscaleObservation(env)
env = FrameStackObservation(env, 4)
env = MaxAndSkipEnv(env, skip=4)

import time
obs, _ = env.reset()
t0 = time.time()
for i in range(10000):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
t = time.time() - t0

print("steps/sec:", 10000/t, "sec per 1M steps:", (1e6/t))
env.close()
