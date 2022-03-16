from gym.envs.registration import register

register(
    id="taichi-v0",
    entry_point="gym_taichi.envs:Taichi_v0",
)

register(
    id="fail-v1",
    entry_point="gym_example2.envs:Fail_v1",
)
