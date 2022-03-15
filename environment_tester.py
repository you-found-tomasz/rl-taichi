from gym_taichi.envs.taichi_env import Taichi_v0
import gym

select_env = "taichi-v0"

env = gym.make(select_env)

state = env.reset()
sum_reward = 0
n_step = 20

for step in range(n_step):
    action = 1
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    sum_reward += reward
    env.render()

