"""Env package."""

from hyperfqi.env.venvs import (
    BaseVectorEnv,
    DummyVectorEnv,
    ShmemVectorEnv,
    SubprocVectorEnv,
)

__all__ = [
    "BaseVectorEnv",
    "DummyVectorEnv",
    "SubprocVectorEnv",
    "ShmemVectorEnv",
]


from gym.envs.registration import register

register(
    id='DeepSea-v0',
    entry_point='hyperfqi.env.deepsea:DeepSea',
)
