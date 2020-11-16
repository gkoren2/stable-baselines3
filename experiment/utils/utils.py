import argparse
import glob
import importlib
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import gym
import yaml
from stable_baselines3 import A2C, DDPG, DQN, HER, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common import logger
try:
    from sb3_contrib import TQC  # pytype: disable=import-error
except ImportError:
    TQC = None

# For custom activation fn
from torch import nn as nn  # noqa: F401 pylint: disable=unused-import

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "her": HER,
    "sac": SAC,
    "td3": TD3,
}

if TQC is not None:
    ALGOS["tqc"] = TQC

def title(msg,n,ch='='):
    return "\n\n"+ch*n+" "+msg+" "+ch*n

def flatten_dict_observations(env: gym.Env) -> gym.Env:
    assert isinstance(env.observation_space, gym.spaces.Dict)
    try:
        return gym.wrappers.FlattenObservation(env)
    except AttributeError:
        keys = env.observation_space.spaces.keys()
        return gym.wrappers.FlattenDictWrapper(env, dict_keys=list(keys))


def get_wrapper_class(hyperparams: Dict[str, Any]) -> Optional[Callable[[gym.Env], gym.Env]]:
    """
    Get one or more Gym environment wrapper class specified as a hyper parameter
    "env_wrapper".
    e.g.
    env_wrapper: gym_minigrid.wrappers.FlatObsWrapper

    for multiple, specify a list:

    env_wrapper:
        - utils.wrappers.PlotActionWrapper
        - utils.wrappers.TimeFeatureWrapper


    :param hyperparams:
    :return: maybe a callable to wrap the environment
        with one or multiple gym.Wrapper
    """

    def get_module_name(wrapper_name):
        return ".".join(wrapper_name.split(".")[:-1])

    def get_class_name(wrapper_name):
        return wrapper_name.split(".")[-1]

    if "env_wrapper" in hyperparams.keys():
        wrapper_name = hyperparams.get("env_wrapper")

        if wrapper_name is None:
            return None

        if not isinstance(wrapper_name, list):
            wrapper_names = [wrapper_name]
        else:
            wrapper_names = wrapper_name

        wrapper_classes = []
        wrapper_kwargs = []
        # Handle multiple wrappers
        for wrapper_name in wrapper_names:
            # Handle keyword arguments
            if isinstance(wrapper_name, dict):
                assert len(wrapper_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {wrapper_name}. "
                    "You should check the indentation."
                )
                wrapper_dict = wrapper_name
                wrapper_name = list(wrapper_dict.keys())[0]
                kwargs = wrapper_dict[wrapper_name]
            else:
                kwargs = {}
            wrapper_module = importlib.import_module(get_module_name(wrapper_name))
            wrapper_class = getattr(wrapper_module, get_class_name(wrapper_name))
            wrapper_classes.append(wrapper_class)
            wrapper_kwargs.append(kwargs)

        def wrap_env(env: gym.Env) -> gym.Env:
            """
            :param env:
            :return:
            """
            for wrapper_class, kwargs in zip(wrapper_classes, wrapper_kwargs):
                env = wrapper_class(env, **kwargs)
            return env

        return wrap_env
    else:
        return None


def get_callback_list(hyperparams: Dict[str, Any]) -> List[BaseCallback]:
    """
    Get one or more Callback class specified as a hyper-parameter
    "callback".
    e.g.
    callback: stable_baselines3.common.callbacks.CheckpointCallback

    for multiple, specify a list:

    callback:
        - utils.callbacks.PlotActionWrapper
        - stable_baselines3.common.callbacks.CheckpointCallback

    :param hyperparams:
    :return:
    """

    def get_module_name(callback_name):
        return ".".join(callback_name.split(".")[:-1])

    def get_class_name(callback_name):
        return callback_name.split(".")[-1]

    callbacks = []

    if "callback" in hyperparams.keys():
        callback_name = hyperparams.get("callback")

        if callback_name is None:
            return callbacks

        if not isinstance(callback_name, list):
            callback_names = [callback_name]
        else:
            callback_names = callback_name

        # Handle multiple wrappers
        for callback_name in callback_names:
            # Handle keyword arguments
            if isinstance(callback_name, dict):
                assert len(callback_name) == 1, (
                    "You have an error in the formatting "
                    f"of your YAML file near {callback_name}. "
                    "You should check the indentation."
                )
                callback_dict = callback_name
                callback_name = list(callback_dict.keys())[0]
                kwargs = callback_dict[callback_name]
            else:
                kwargs = {}
            callback_module = importlib.import_module(get_module_name(callback_name))
            callback_class = getattr(callback_module, get_class_name(callback_name))
            callbacks.append(callback_class(**kwargs))

    return callbacks


def create_test_env(
    env_id: str,
    n_envs: int = 1,
    stats_path: Optional[str] = None,
    seed: int = 0,
    log_dir: Optional[str] = None,
    should_render: bool = True,
    hyperparams: Optional[Dict[str, Any]] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> VecEnv:
    """
    Create environment for testing a trained agent

    :param env_id:
    :param n_envs: number of processes
    :param stats_path: path to folder containing saved running averaged
    :param seed: Seed for random number generator
    :param log_dir: Where to log rewards
    :param should_render: For Pybullet env, display the GUI
    :param hyperparams: Additional hyperparams (ex: n_stack)
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :return:
    """
    # Create the environment and wrap it if necessary
    env_wrapper = get_wrapper_class(hyperparams)

    hyperparams = {} if hyperparams is None else hyperparams

    if "env_wrapper" in hyperparams.keys():
        del hyperparams["env_wrapper"]

    vec_env_kwargs = {}
    vec_env_cls = DummyVecEnv
    if n_envs > 1 or "Bullet" in env_id:
        # HACK: force SubprocVecEnv for Bullet env
        # as Pybullet envs does not follow gym.render() interface
        vec_env_cls = SubprocVecEnv
        # start_method = 'spawn' for thread safe

    env = make_vec_env(
        env_id,
        n_envs=n_envs,
        monitor_dir=log_dir,
        seed=seed,
        wrapper_class=env_wrapper,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )

    # Load saved stats for normalizing input and rewards
    # And optionally stack frames
    if stats_path is not None:
        if hyperparams["normalize"]:
            print("Loading running average")
            print(f"with params: {hyperparams['normalize_kwargs']}")
            path_ = os.path.join(stats_path, "vecnormalize.pkl")
            if os.path.exists(path_):
                env = VecNormalize.load(path_, env)
                # Deactivate training and reward normalization
                env.training = False
                env.norm_reward = False
            else:
                raise ValueError(f"VecNormalize stats {path_} not found")

        n_stack = hyperparams.get("frame_stack", 0)
        if n_stack > 0:
            print(f"Stacking {n_stack} frames")
            env = VecFrameStack(env, n_stack)
    return env


def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def get_trained_models(log_folder: str) -> Dict[str, Tuple[str, str]]:
    """
    :param log_folder: (str) Root log folder
    :return: (Dict[str, Tuple[str, str]]) Dict representing the trained agent
    """
    trained_models = {}
    for algo in os.listdir(log_folder):
        if not os.path.isdir(os.path.join(log_folder, algo)):
            continue
        for env_id in os.listdir(os.path.join(log_folder, algo)):
            # Retrieve env name
            env_id = env_id.split("_")[0]
            trained_models[f"{algo}-{env_id}"] = (algo, env_id)
    return trained_models


def get_latest_run_id(log_path: str, env_id: str) -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :param env_id:
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(log_path + f"/{env_id}_[0-9]*"):
        file_name = path.split("/")[-1]
        ext = file_name.split("_")[-1]
        if env_id == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def get_saved_hyperparams(stats_path: str, norm_reward: bool = False, test_mode: bool = False) -> Tuple[Dict[str, Any], str]:
    """
    :param stats_path:
    :param norm_reward:
    :param test_mode:
    :return:
    """
    hyperparams = {}
    if not os.path.isdir(stats_path):
        stats_path = None
    else:
        config_file = os.path.join(stats_path, "config.yml")
        if os.path.isfile(config_file):
            # Load saved hyperparameters
            with open(os.path.join(stats_path, "config.yml"), "r") as f:
                hyperparams = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
            hyperparams["normalize"] = hyperparams.get("normalize", False)
        else:
            obs_rms_path = os.path.join(stats_path, "obs_rms.pkl")
            hyperparams["normalize"] = os.path.isfile(obs_rms_path)

        # Load normalization params
        if hyperparams["normalize"]:
            if isinstance(hyperparams["normalize"], str):
                normalize_kwargs = eval(hyperparams["normalize"])
                if test_mode:
                    normalize_kwargs["norm_reward"] = norm_reward
            else:
                normalize_kwargs = {"norm_obs": hyperparams["normalize"], "norm_reward": norm_reward}
            hyperparams["normalize_kwargs"] = normalize_kwargs
    return hyperparams, stats_path


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)


def online_eval_results_analysis(npz_filename):
    if not os.path.exists(npz_filename):
        logger.warn('evaluation results file not found')
        return

    eval_np = np.load(npz_filename)
    eval_dict = {k:v for k,v in eval_np.items()}
    # results include all results for all epochs. the shape is [T,n_epochs,1]
    # we want to generate statistics for the epochs of each time step:
    # replace 'results' with 'mean_rew' and 'std_rew':
    eval_dict['reward_mean'] = np.squeeze(eval_dict['results']).mean(axis=1)
    eval_dict['reward_std'] = np.squeeze(eval_dict['results']).std(axis=1)
    del eval_dict['results']
    # same goes for the ep_lengths field:
    eval_dict['ep_lengths_mean'] = eval_dict['ep_lengths'].mean(axis=1)
    eval_dict['ep_lengths_std']= eval_dict['ep_lengths'].std(axis=1)
    del eval_dict['ep_lengths']
    eval_df = pd.DataFrame(eval_dict)
    eval_df.set_index('timesteps',inplace=True)
    # save csv filename
    # df_filename = os.path.splitext(npz_filename)[0]+'.csv'
    # eval_df.to_csv(df_filename)
    return eval_df

def generate_experience_traj(model, save_path=None, env=None, n_timesteps_train=0,
                         n_timesteps_record=100000,deterministic=True,with_prob=True):
    """
    Train expert controller (if needed) and record expert trajectories.

    .. note::

        only Box and Discrete spaces are supported for now.
        support in images is removed.

    :param model: (RL model or callable) The expert model, if it needs to be trained,
        then you need to pass ``n_timesteps > 0``.
        note that the RL model can be also a pretrained expert that was loaded from file.
    :param save_path: (str) Path without the extension where the expert dataset will be saved
        (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
        If not specified, it will not save, and just return the generated expert trajectories.
        This parameter must be specified for image-based environments.
    :param env: (gym.Env) The environment, if not defined then it tries to use the model
        environment.
    :param n_timesteps_train: (int) Number of training timesteps
    :param n_timesteps_record: (int) Number of trajectories (episodes) to record
    :param logger: (Logger) - if not None, use it for verbose output
    :return: (dict) the generated expert trajectories.
    """

    # Retrieve the environment using the RL model
    if env is None and isinstance(model, BaseRLModel):
        env = model.get_env()

    assert env is not None, "You must set the env in the model or pass it to the function."

    is_vec_env = False
    if isinstance(env, VecEnv) and not isinstance(env, _UnvecWrapper):
        is_vec_env = True
        if env.num_envs > 1:
            warnings.warn("You are using multiple envs, only the data from the first one will be recorded.")

    # Sanity check
    assert (isinstance(env.observation_space, spaces.Box) or
            isinstance(env.observation_space, spaces.Discrete)), "Observation space type not supported"

    assert (isinstance(env.action_space, spaces.Box) or
            isinstance(env.action_space, spaces.Discrete)), "Action space type not supported"

    # Note: support in recording image to files is omitted
    obs_space = env.observation_space
    replay_buffer = ReplayBuffer(n_timesteps_record)

    logger.info(title("generate expert trajectory",20))

    if n_timesteps_train > 0 and isinstance(model, BaseRLModel):
        logger.info("training expert start - {0} timesteps".format(n_timesteps_train))
        model.learn(n_timesteps_train,tb_log_name='exp_gen_train')
        logger.info("generate expert trajectory: training expert end")


    logger.info("start recording {0} expert steps".format(n_timesteps_record))
    episode_returns = []
    episode_starts = []

    ep_idx = 0
    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0
    # state and mask for recurrent policies
    state, mask = None, None
    if is_vec_env:
        mask = [True for _ in range(env.num_envs)]
    for t in tqdm(range(n_timesteps_record)):
        info={}
        if isinstance(model, BaseRLModel):
            if with_prob:
                action, state, act_prob = model.predict(obs, state=state, mask=mask,deterministic=deterministic,with_prob=True)
                # info.update({'all_action_probabilities': str(act_prob)})
                info.update({'all_action_probabilities': act_prob})
            else:       # default is with_prob=False
                action, state = model.predict(obs, state=state, mask=mask,deterministic=deterministic)
        else:   # random agent that samples uniformly
            if with_prob:
                assert isinstance(env.action_space,gym.spaces.Discrete), "currently supporting action prob in Discrete space only"
                action,act_prob = model.predict(obs,with_prob=True)
                # info.update({'all_action_probabilities': str(act_prob)})
                info.update({'all_action_probabilities': act_prob})
            else:
                action = model.predict(obs)

        new_obs, reward, done, _ = env.step(action)

        # Note : we save to the experience buffer as if it is not a vectorized env since anyway we
        #        use only first env

        if is_vec_env:
            mask = [done[0] for _ in range(env.num_envs)]
            action = np.array([action[0]])
            reward = np.array(reward[0])
            done = np.array([done[0]])
            info = np.array([info[0]])
            replay_buffer.add(obs[0],action[0],reward,new_obs[0],float(done[0]),info[0])
        else: # Store transition in the replay buffer.
            replay_buffer.add(obs, action, reward, new_obs, float(done),info)
        obs = new_obs
        episode_starts.append(done)
        reward_sum += reward
        if done:
            if not is_vec_env:
                obs = env.reset()
                # Reset the state in case of a recurrent policy
                state = None
            episode_returns.append(reward_sum)
            reward_sum = 0.0
            ep_idx += 1

    logger.info("finished collecting experience data")
    numpy_dict = replay_buffer.record_buffer()
    # Note : the ReplayBuffer can not generally assume it has not circled around thus cant infer accurate episode
    # statistics. since in this context we know these details, we overwrite the corresponding fields:
    numpy_dict['episode_returns'] = np.array(episode_returns)
    numpy_dict['episode_starts'] = np.array(episode_starts[:-1])
    logger.info("Statistics: {0} episodes, average return={1}".format(len(episode_returns),
                                                                      np.mean(numpy_dict['episode_returns'])))
    # assuming we save only the numpy arrays (not the obs_space and act_space)
    if save_path is not None:
        np.savez(save_path+'.npz', **numpy_dict)
        logger.info("saving to experience as csv file: "+save_path+'.csv')
        replay_buffer.save_to_csv(save_path+'.csv',os.path.splitext(os.path.basename(save_path))[0])

    env.close()
    return numpy_dict
