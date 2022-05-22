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
sum_reward = 0
n_step = 10000


#config = ppo.DEFAULT_CONFIG.copy()
#agent = ppo.PPOTrainer(config, env=select_env)
#agent.restore('tmp/exa/checkpoint_000023/checkpoint-23')

for step in range(n_step):
    action = env.action_space.sample()
    #action = agent.compute_action(state)
    state, reward, done, info = env.step(action)
    if info['cloth'] == True:
        #print("cloth broken")
        state = env.reset()
        #np.save('finished_state.npy', env.previous_state_full)
        break
    sum_reward += reward
    env.render(action)

