#!/usr/bin/env python3

from gym.envs.registration import register

from .subproc_vec_env import SubprocVecEnv

# 2D Navigation
# ----------------------------------------

register(
    'Particles2D-v1',
    entry_point='MetaRL.gym.envs.particles.particles_2d:Particles2DEnv',
    max_episode_steps=100
)

register(
    'Deeproute-stat-v0',
    entry_point='MetaRL.gym.envs.deeproute.deeproute_stat_env:DeeprouteStatEnv',
    max_episode_steps=1000
)