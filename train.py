#!/usr/bin/env python
# encoding: utf-8

from gym_taichi.envs.taichi_env import Taichi_v0
from ray.tune.registry import register_env
import gym
import os
import ray
import ray.rllib.agents.ppo as ppo
import shutil


def main ():
    # init directory in which to save checkpoints
    chkpt_root = "tmp/exa"
    shutil.rmtree(chkpt_root, ignore_errors=True, onerror=None)

    # init directory in which to log results
    ray_results = "{}/ray_results/".format(os.getenv("HOME"))
    shutil.rmtree(ray_results, ignore_errors=True, onerror=None)

    # start Ray -- add `local_mode=True` here for debugging
    ray.init(ignore_reinit_error=True, include_dashboard=True, logging_level=3, object_store_memory=78643200, _redis_max_memory=78643200, _memory=78643200, num_gpus=1)#, local_mode=True)

    # register the custom environment
    select_env = "taichi-v0"
    #select_env = "CartPole-v0"
    #select_env = "fail-v1"
    register_env(select_env, lambda config: Taichi_v0())
    #register_env(select_env, lambda config: CartPole_v0())
    #register_env(select_env, lambda config: Fail_v1())


    # configure the environment and create agent
    config = ppo.DEFAULT_CONFIG.copy()
    config["log_level"] = "WARN"
    config["num_cpus_per_worker"] = 1
    config["num_workers"] = 1
    config["framework"] = "tf"
    config["train_batch_size"] = 4000
    config["num_gpus"] = 0
    config["num_gpus_per_worker"] = 0
    #config["reuse_actors"] = True
    config["output_max_file_size"] = 500000
    #config["buffer_size"] = 10000
    config["batch_mode"] = "truncate_episodes"
    config["num_envs_per_worker"] = 1
    config["ignore_worker_failures"] = True
    config["disable_env_checking"] = True

    agent = ppo.PPOTrainer(config, env=select_env)

    status = "{:2d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:4.2f} saved {}"
    n_iter = 100

    # train a policy with RLlib using PPO
    for n in range(n_iter):
        if n == 0:
            result = agent.train()
            chkpt_file = agent.save(chkpt_root)
        else:
            agent = ppo.PPOTrainer(config, env=select_env)
            agent.restore(chkpt_file)
            result = agent.train()
            chkpt_file = agent.save(chkpt_root)
        agent.stop()

        print(status.format(
                n + 1,
                result["episode_reward_min"],
                result["episode_reward_mean"],
                result["episode_reward_max"],
                result["episode_len_mean"],
                chkpt_file
                ))

    # examine the trained policy
    policy = agent.get_policy()
    model = policy.model
    print(model.summary())


    # apply the trained policy in a rollout
    agent.restore(chkpt_file)
    env = gym.make(select_env)

    state = env.reset()
    sum_reward = 0
    n_step = 10

    for step in range(n_step):
        action = agent.compute_action(state)
        state, reward, done, info = env.step(action)
        sum_reward += reward
        env.render(action)

        if info['cloth'] == True:
            #print("cloth broken")
            state = env.reset()
            break

        if done == 1:
            # report at the end of each episode
            print("cumulative reward", sum_reward)
            state = env.reset()
            sum_reward = 0


if __name__ == "__main__":
    main()
