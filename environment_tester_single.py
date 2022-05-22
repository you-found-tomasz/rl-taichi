from gym_taichi.envs.taichi_env import Taichi_v0
from particle_simulator_wrapper import Particle_Simulator
import ray.rllib.agents.ppo as ppo
from ray.tune.registry import register_env
import numpy as np
import gym
# working version taichi                       0.8.11

select_env = "taichi-v0"
register_env(select_env, lambda config: Taichi_v0())
env = gym.make(select_env)
state = env.reset()
loaded_state = np.load('finished_state.npy')
env.state = loaded_state
env.simulator.simulate(env.state)