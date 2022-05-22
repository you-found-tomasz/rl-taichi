import gym
from particle_simulator_wrapper import Particle_Simulator
simulator = Particle_Simulator()

k = 200
for i in range(100):
    simulator.update(k+i,0)
simulator.update(341,0)
simulator.update(611,0)

simulator.simulate()
simulator.simulate()
print("end")
from gym_taichi.envs.taichi_env import Taichi_v0
from ray.tune.registry import register_env

select_env = "taichi-v0"
register_env(select_env, lambda config: Taichi_v0())
env = gym.make('taichi-v0')

t = 0
observation = env.reset()
while True:
   t += 1
   #env.render()
   #print(observation)
   action = env.action_space.sample()
   observation, reward, done, info = env.step(action)
   print(observation, reward, done)
   if done:
       print("Episode finished after {} timesteps".format(t+1))
       break
env.close()