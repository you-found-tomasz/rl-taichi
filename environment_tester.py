from gym_taichi.envs.taichi_env import Taichi_v0
from particle_simulator_wrapper import Particle_Simulator
import gym
# working version taichi                       0.8.11

select_env = "taichi-v0"

env = gym.make(select_env)

state = env.reset()
sum_reward = 0
n_step = 200
#simulator = Particle_Simulator()
#simulator.simulate([])
#simulator.simulate([])

for step in range(n_step):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if info['cloth'] == True:
        print("cloth broken")
        state = env.reset()
        break
    sum_reward += reward
    env.render()

