try:
    import pybullet_envs  # pytype: disable=import-error
except ImportError:
    pybullet_envs = None

try:
    import highway_env  # pytype: disable=import-error
except ImportError:
    highway_env = None

try:
    import neck_rl  # pytype: disable=import-error
except ImportError:
    neck_rl = None

try:
    import mocca_envs  # pytype: disable=import-error
except ImportError:
    mocca_envs = None

try:
    import custom_envs  # pytype: disable=import-error
except ImportError:
    custom_envs = None

try:
    import gym_donkeycar  # pytype: disable=import-error
except ImportError:
    gym_donkeycar = None

CC_ENVS = {'cartpole':'CartPole-v0',
           'cartpole1':'CartPole-v1',
           'mntcar':'MountainCar-v0',
           'acrobot':'Acrobot-v1',
           'lunland':'LunarLander-v2'
           }
