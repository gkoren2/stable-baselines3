import argparse
import csv
import json
import os
import time
import warnings
from collections import OrderedDict
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple

import gym
import numpy as np
import optuna
import yaml
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler
from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike  # noqa: F401
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack, VecNormalize, VecTransposeImage
from stable_baselines3.common.vec_env.obs_dict_wrapper import ObsDictWrapper

# For custom activation fn
from torch import nn as nn  # noqa: F401

# Register custom envs
import experiment.envs.import_envs  # noqa: F401 pytype: disable=import-error
from experiment.utils.callbacks import SaveVecNormalizeCallback, TrialEvalCallback, OnlEvalTBCallback
from experiment.utils.hyperparams_opt import HYPERPARAMS_SAMPLER
from experiment.utils.utils import ALGOS, get_callback_list, get_latest_run_id, get_wrapper_class, linear_schedule,\
    online_eval_results_analysis,generate_experience_traj




class ExperimentManager(object):
    """
    Experiment manager: read the hyperparameters,
    preprocess them, create the environment and the RL model.

    """
    def __init__(self,params):
        super(ExperimentManager, self).__init__()
        self.experiment_params = params

        # extract the experiment params
        self.verbose = self.experiment_params.verbose
        self.output_dir = os.path.join(self.experiment_params.output_root_dir,str(self.experiment_params.seed))

        self.algo = self.experiment_params.agent_params.algorithm
        env_id = self.experiment_params.env_params.env_id
        self.env_id = experiment.envs.import_envs.CC_ENVS.get(env_id,env_id)
        # Custom params
        self.custom_hyperparams = None
        self.env_kwargs = self.experiment_params.env_params.env_kwargs
        self.n_timesteps = self.experiment_params.n_timesteps
        self.normalize = False      # will be extracted from env_params
        self.normalize_kwargs = {}
        self.env_wrapper = None
        self.frame_stack = None
        self.seed = self.experiment_params.seed

        # Callbacks
        self.callbacks = []
        self.save_freq = int(self.experiment_params.checkpoint_save_freq)
        self.eval_freq = int(self.experiment_params.online_eval_freq)
        self.n_eval_episodes = self.experiment_params.online_eval_n_episodes

        self.n_envs = 1  # it will be updated when reading hyperparams
        self.n_actions = None  # For DDPG/TD3 action noise objects
        self._hyperparams = {}

        self.trained_agent = self.experiment_params.trained_agent_model_file or ""
        self.continue_training = self.trained_agent.endswith(".zip") and os.path.isfile(self.trained_agent)
        self.truncate_last_trajectory = self.experiment_params.truncate_last_trajectory

        self._is_atari = self.is_atari(self.env_id)
        # Hyperparameter optimization config
        self.optimize_hyperparameters = self.experiment_params.hpopt_on
        # Database storage path if distributed optimization should be used
        self.storage = self.experiment_params.hpopt_storage
        # Study name for distributed optimization
        self.study_name = self.experiment_params.hpopt_study_name
        # maximum number of trials for finding the best hyperparams
        self.n_trials = self.experiment_params.hpopt_n_trials
        # number of parallel jobs when doing hyperparameter search
        self.n_jobs = self.experiment_params.hpopt_n_jobs
        # Sampler to use when optimizing hyperparameters (["random", "tpe", "skopt"])
        self.sampler = self.experiment_params.hpopt_sampler
        # Pruner to use when optimizing hyperparameters
        self.pruner = self.experiment_params.hpopt_pruner
        # Number of trials before using optuna sampler
        self.n_startup_trials = self.experiment_params.hpopt_n_startup_trials
        # Number of evaluations for hyperparameter optimization
        self.n_evaluations = self.experiment_params.hpopt_n_evaluations
        self.deterministic_eval = not self._is_atari

        # number of steps to record from the expert model
        self.expert_steps_to_record = self.experiment_params.expert_steps_to_record

        # Logging
        self.log_folder = self.output_dir
        self.tensorboard_log = self.output_dir if self.experiment_params.log_tensorboard else None
        self.verbose = self.experiment_params.verbose
        self.args = None
        self.log_interval = self.experiment_params.log_interval
        self.save_replay_buffer = self.experiment_params.save_replay_buffer

        self.log_path = self.output_dir
        self.save_path = self.output_dir
        self.params_path = self.output_dir



    def setup_experiment(self) -> Optional[BaseAlgorithm]:
        """
        Read hyperparameters, pre-process them (create schedules, wrappers, callbacks, action noise objects)
        create the environment and possibly the model.

        :return: the initialized RL model
        """
        self.create_log_folder()


        if self.n_timesteps <= 0:
            # imitate what is done in enjoy
            # prepare for environment creation
            env_params = self.experiment_params.env_params.as_dict()
            self.env_wrapper = self._preprocess_env_params(env_params)
            # Create env to have access to action space for action noise
            self.n_envs = self.experiment_params.n_envs
            env = self.create_envs(self.n_envs, eval_env=True)
            # load the model
            if self.algo=='random':
                model = ALGOS[self.algo](env)
            else:
                off_policy_algos = ["dqn", "ddpg", "sac", "her", "td3", "tqc"]
                kwargs = dict(seed=self.seed)
                if self.algo in off_policy_algos:
                    # Dummy buffer size as we don't need memory to enjoy the trained agent
                    kwargs.update(dict(buffer_size=1))
                model = self._load_pretrained_agent(kwargs,env)
        else:       # model needs to be trained or retrained
            agent_hyperparams, env_params, saved_hyperparams = self.read_hyperparameters()
            agent_hyperparams, self.callbacks = self._preprocess_agent_hyperparams(agent_hyperparams)

            self.env_wrapper = self._preprocess_env_params(env_params)

            self.create_callbacks()

            # Create env to have access to action space for action noise
            self.n_envs = self.experiment_params.n_envs
            env = self.create_envs(self.n_envs, no_log=False)

            self._hyperparams = self._preprocess_action_noise(agent_hyperparams, env)

            if self.continue_training:
                model = self._load_pretrained_agent(self._hyperparams, env)
            elif self.optimize_hyperparameters:
                return None
            else:
                # Train an agent from scratch
                model = ALGOS[self.algo](env=env, **self._hyperparams)

            self._save_config(saved_hyperparams)
        return model


    def learn(self, model: BaseAlgorithm) -> None:
        """
        :param model: an initialized RL model
        """
        kwargs = {}
        if self.log_interval > -1:
            kwargs = {"log_interval": self.log_interval}

        if len(self.callbacks) > 0:
            kwargs["callback"] = self.callbacks

        try:
            model.learn(self.n_timesteps, **kwargs)
            if self.n_eval_episodes>0:
                online_eval_results_analysis(os.path.join(self.log_path, 'evaluations.npz'))

        except KeyboardInterrupt:
            # this allows to save the model when interrupting training
            pass
        finally:
            # Release resources
            model.env.close()

    def save_trained_model(self, model: BaseAlgorithm) -> None:
        """
        Save trained model optionally with its replay buffer
        and ``VecNormalize`` statistics

        :param model:
        """
        model_file_name = os.path.join(self.save_path,'final_model')
        logger.info(f"Saving to {model_file_name}")
        model.save(model_file_name)

        if hasattr(model, "save_replay_buffer") and self.save_replay_buffer:
            logger.info("Saving replay buffer")
            model.save_replay_buffer(os.path.join(self.save_path, "replay_buffer.pkl"))

        if self.normalize:
            # Important: save the running average, for testing the agent we need that normalization
            model.get_vec_normalize_env().save(os.path.join(self.params_path, "vecnormalize.pkl"))

    def rollout_on_env(self,model: BaseAlgorithm, experience_file_name:str ="") -> None:
        if experience_file_name=="":
            experience_file_name = 'er_'+self.env_id+'_'+self.algo+'_'+str(self.expert_steps_to_record)
        experience_file_name = os.path.join(self.save_path,experience_file_name)
        # if self.env_id=='DTTSim':
        #     self.experiment_params.env_params.log_output = self.save_path
        if self.n_timesteps>0:      # need to define eval env as the model.env was training env
            eval_env = self.create_envs(1,eval_env=True)
            model.set_env(eval_env)
        logger.info('Generating expert experience buffer with ' + self.algo)
        _ = generate_experience_traj(model, save_path=experience_file_name,
                                     n_timesteps_record=self.experiment_params.expert_steps_to_record)
        return



    def read_hyperparameters(self) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:

        # take a snapshot of the hyper parameters to save
        agent_hyperparams = self.experiment_params.agent_params.as_dict()
        env_params_dict = self.experiment_params.env_params.as_dict()
        exparams_dict = self.experiment_params.as_dict()
        saved_env_params = OrderedDict([(key, str(env_params_dict[key])) for key in sorted(env_params_dict.keys())])
        saved_agent_hyperparams = OrderedDict(
            [(key, str(agent_hyperparams[key])) for key in sorted(agent_hyperparams.keys())])

        saved_hyperparams = OrderedDict([(key, str(exparams_dict[key])) for key in exparams_dict.keys()])
        saved_hyperparams['agent_params'] = saved_agent_hyperparams
        saved_hyperparams['env_params'] = saved_env_params

        return agent_hyperparams, env_params_dict,saved_hyperparams


    def create_log_folder(self):
        os.makedirs(self.output_dir, exist_ok=True)


    def _preprocess_env_params(self,env_params:Dict[str,Any]) -> Optional[Callable]:
        self.frame_stack = env_params['frame_stack']
        env_wrapper = get_wrapper_class(env_params)

        return env_wrapper


    def _preprocess_agent_hyperparams(
        self, agent_hyperparams: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[BaseCallback]]:

        # Convert model class string to an object if needed (when using HER)
        agent_hyperparams = self._preprocess_her_model_class(agent_hyperparams)
        agent_hyperparams = self._preprocess_schedules(agent_hyperparams)

        # Pre-process normalize config
        self._preprocess_normalization(agent_hyperparams)
        # hyperparams = self._preprocess_normalization(hyperparams)

        # Pre-process policy keyword arguments
        if "policy_kwargs" in agent_hyperparams.keys():
            # Convert to python object if needed
            if isinstance(agent_hyperparams["policy_kwargs"], str):
                agent_hyperparams["policy_kwargs"] = eval(agent_hyperparams["policy_kwargs"])

        callbacks = get_callback_list(agent_hyperparams)
        if "callback" in agent_hyperparams.keys():
            del agent_hyperparams["callback"]

        agent_hyperparams['tensorboard_log'] = self.tensorboard_log

        del agent_hyperparams['algorithm']

        return agent_hyperparams, callbacks

    def create_envs(self, n_envs: int, eval_env: bool = False, no_log: bool = False) -> VecEnv:
        """
        Create the environment and wrap it if necessary.

        :param n_envs:
        :param eval_env: Whether is it an environment used for evaluation or not
        :param no_log: Do not log training when doing hyperparameter optim
            (issue with writing the same file)
        :return: the vectorized environment, with appropriate wrappers
        """
        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env or no_log else self.output_dir

        # env = SubprocVecEnv([make_env(env_id, i, self.seed) for i in range(n_envs)])
        # On most env, SubprocVecEnv does not help and is quite memory hungry
        env = make_vec_env(
            env_id=self.env_id,
            n_envs=n_envs,
            seed=self.experiment_params.seed,
            env_kwargs=self.experiment_params.env_params.env_kwargs,
            monitor_dir=log_dir,
            wrapper_class=self.env_wrapper,
            vec_env_cls=DummyVecEnv,
            vec_env_kwargs=None,
        )

        # Special case for GoalEnvs: log success rate too
        if "Neck" in self.env_id or self.is_robotics_env(self.env_id):
            self._log_success_rate(env)

        # Wrap the env into a VecNormalize wrapper if needed
        # and load saved statistics when present
        env = self._maybe_normalize(env, eval_env)

        # Optional Frame-stacking
        if self.frame_stack is not None:
            n_stack = self.frame_stack
            env = VecFrameStack(env, n_stack)
            if self.verbose > 0:
                logger.info(f"Stacking {n_stack} frames")

        # Wrap if needed to re-order channels
        # (switch from channel last to channel first convention)
        if is_image_space(env.observation_space):
            if self.verbose > 0:
                logger.info("Wrapping into a VecTransposeImage")
            env = VecTransposeImage(env)

        # check if wrapper for dict support is needed
        if self.algo == "her":
            if self.verbose > 0:
                logger.info("Wrapping into a ObsDictWrapper")
            env = ObsDictWrapper(env)

        return env

    def _maybe_normalize(self, env: VecEnv, eval_env: bool) -> VecEnv:
        """
        Wrap the env into a VecNormalize wrapper if needed
        and load saved statistics when present.

        :param env:
        :param eval_env:
        :return:
        """
        # Pretrained model, load normalization
        path_ = os.path.join(os.path.dirname(self.trained_agent), self.env_id)
        path_ = os.path.join(path_, "vecnormalize.pkl")

        if os.path.exists(path_):
            logger.info("Loading saved VecNormalize stats")
            env = VecNormalize.load(path_, env)
            # Deactivate training and reward normalization
            if eval_env:
                env.training = False
                env.norm_reward = False

        elif self.normalize:
            # Copy to avoid changing default values by reference
            local_normalize_kwargs = self.normalize_kwargs.copy()
            # Do not normalize reward for env used for evaluation
            if eval_env:
                if len(local_normalize_kwargs) > 0:
                    local_normalize_kwargs["norm_reward"] = False
                else:
                    local_normalize_kwargs = {"norm_reward": False}

            if self.verbose > 0:
                if len(local_normalize_kwargs) > 0:
                    logger.info(f"Normalization activated: {local_normalize_kwargs}")
                else:
                    logger.info("Normalizing input and reward")
            env = VecNormalize(env, **local_normalize_kwargs)
        return env


    def _preprocess_normalization(self,agent_hyperparams: Dict[str, Any]) -> None:
        norm_obs = self.experiment_params.env_params.norm_obs
        norm_reward = self.experiment_params.env_params.norm_reward
        self.normalize = norm_obs or norm_reward
        if self.normalize:
            # Special case, instead of both normalizing
            # both observation and reward, we can normalize one of the two.
            # in that case `hyperparams["normalize"]` is a string
            # that can be evaluated as python,
            # ex: "dict(norm_obs=False, norm_reward=True)"
            self.normalize_kwargs=dict(norm_obs=norm_obs,norm_reward=norm_reward)
            # Use the same discount factor as for the algorithm
            if "gamma" in agent_hyperparams:
                self.normalize_kwargs["gamma"] = agent_hyperparams["gamma"]
        return

    def _preprocess_her_model_class(self, agent_hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        # HER is only a wrapper around an algo
        if self.algo == "her":
            model_class = agent_hyperparams["model_class"]
            assert model_class in {"sac", "ddpg", "dqn", "td3", "tqc"}, f"{model_class} is not compatible with HER"
            # Retrieve the model class
            agent_hyperparams["model_class"] = ALGOS[agent_hyperparams["model_class"]]
        return agent_hyperparams

    @staticmethod
    def _preprocess_schedules(hyperparams: Dict[str, Any]) -> Dict[str, Any]:
        # Create schedules
        for key in ["learning_rate", "clip_range", "clip_range_vf"]:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split("_")
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constant_fn(float(hyperparams[key]))
            else:
                raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
        return hyperparams

    def create_callbacks(self):

        if self.save_freq > 0:
            # Account for the number of parallel environments
            self.save_freq = max(self.save_freq // self.n_envs, 1)
            self.callbacks.append(
                CheckpointCallback(
                    save_freq=self.save_freq,
                    save_path=self.save_path,
                    name_prefix="rl_model",
                    verbose=1,
                )
            )

        # Create test env if needed, do not normalize reward
        if self.eval_freq > 0 and not self.optimize_hyperparameters:
            # Account for the number of parallel environments
            self.eval_freq = max(self.eval_freq // self.n_envs, 1)
            save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=self.params_path)
            eval_callback = OnlEvalTBCallback(
                self.create_envs(1, eval_env=True),
                callback_on_new_best=save_vec_normalize,
                best_model_save_path=self.save_path,
                n_eval_episodes=self.n_eval_episodes,
                log_path=self.save_path,
                eval_freq=self.eval_freq,
                deterministic=self.deterministic_eval
            )

            self.callbacks.append(eval_callback)

    def _preprocess_action_noise(self, agent_hyperparams: Dict[str, Any], env: VecEnv) -> Dict[str, Any]:
        # Special case for HER
        algo = agent_hyperparams["model_class"] if self.algo == "her" else self.algo
        # Parse noise string for DDPG and SAC
        if algo in ["ddpg", "sac", "td3", "tqc", "ddpg"] and agent_hyperparams.get("noise_type") is not None:
            noise_type = agent_hyperparams["noise_type"].strip()
            noise_std = agent_hyperparams["noise_std"]

            # Save for later (hyperparameter optimization)
            self.n_actions = env.action_space.shape[0]

            if "normal" in noise_type:
                agent_hyperparams["action_noise"] = NormalActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            elif "ornstein-uhlenbeck" in noise_type:
                agent_hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(self.n_actions),
                    sigma=noise_std * np.ones(self.n_actions),
                )
            else:
                raise RuntimeError(f'Unknown noise type "{noise_type}"')

            print(f"Applying {noise_type} noise with std {noise_std}")

            del agent_hyperparams["noise_type"]
            del agent_hyperparams["noise_std"]

        return agent_hyperparams


    def _load_pretrained_agent(self, hyperparams: Dict[str, Any], env: VecEnv) -> BaseAlgorithm:
        # Continue training
        logger.info("Loading pretrained agent")
        # Policy should not be changed
        if "policy" in hyperparams.keys():
            del hyperparams["policy"]

        if "policy_kwargs" in hyperparams.keys():
            del hyperparams["policy_kwargs"]

        model = ALGOS[self.algo].load(
            self.trained_agent,
            env=env,
            **hyperparams,
        )

        if self.n_timesteps > 0:        # loading the model to continue training
            replay_buffer_path = os.path.join(os.path.dirname(self.trained_agent), "replay_buffer.pkl")

            if os.path.exists(replay_buffer_path):
                logger.info("Loading replay buffer")
                if self.algo == "her":
                    # if we use HER we have to add an additional argument
                    model.load_replay_buffer(replay_buffer_path, self.truncate_last_trajectory)
                else:
                    model.load_replay_buffer(replay_buffer_path)
        return model


    def _save_config(self, saved_hyperparams: Dict[str, Any]) -> None:
        """
        Save unprocessed hyperparameters, this can be use later
        to reproduce an experiment.

        :param saved_hyperparams:
        """
        # Save hyperparams
        with open(os.path.join(self.params_path, "config.yml"), "w") as f:
            yaml.dump(saved_hyperparams, f)

        # save command line arguments
        # with open(os.path.join(self.params_path, "args.yml"), "w") as f:
        #     ordered_args = OrderedDict([(key, vars(self.args)[key]) for key in sorted(vars(self.args).keys())])
        #     yaml.dump(ordered_args, f)

        logger.info(f"Log path: {self.save_path}")

    @staticmethod
    def is_atari(env_id: str) -> bool:
        return "AtariEnv" in gym.envs.registry.env_specs[env_id].entry_point

    @staticmethod
    def is_robotics_env(env_id: str) -> bool:
        return "gym.envs.robotics" in gym.envs.registry.env_specs[env_id].entry_point


    def _log_success_rate(self, env: VecEnv) -> None:
        # Hack to log the success rate
        # TODO: allow to pass keyword arguments to the Monitor class
        monitor: gym.Env = env.envs[0]
        # unwrap
        while not isinstance(monitor, Monitor):
            monitor = monitor.env

        if monitor.file_handler is None:
            return

        filename = monitor.file_handler.name
        monitor.file_handler.close()

        monitor.info_keywords = ("is_success",)
        monitor.file_handler = open(filename, "wt")
        monitor.file_handler.write(
            "#%s\n" % json.dumps({"t_start": monitor.t_start, "env_id": monitor.env.spec and monitor.env.spec.id})
        )
        monitor.logger = csv.DictWriter(monitor.file_handler, fieldnames=("r", "l", "t") + monitor.info_keywords)
        monitor.logger.writeheader()
        monitor.file_handler.flush()



    def _create_sampler(self, sampler_method: str) -> BaseSampler:
        # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
        if sampler_method == "random":
            sampler = RandomSampler(seed=self.seed)
        elif sampler_method == "tpe":
            # TODO: try with multivariate=True
            sampler = TPESampler(n_startup_trials=self.n_startup_trials, seed=self.seed)
        elif sampler_method == "skopt":
            # cf https://scikit-optimize.github.io/#skopt.Optimizer
            # GP: gaussian process
            # Gradient boosted regression: GBRT
            sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
        else:
            raise ValueError(f"Unknown sampler: {sampler_method}")
        return sampler

    def _create_pruner(self, pruner_method: str) -> BasePruner:
        if pruner_method == "halving":
            pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
        elif pruner_method == "median":
            pruner = MedianPruner(n_startup_trials=self.n_startup_trials, n_warmup_steps=self.n_evaluations // 3)
        elif pruner_method == "none":
            # Do not prune
            pruner = MedianPruner(n_startup_trials=self.n_trials, n_warmup_steps=self.n_evaluations)
        else:
            raise ValueError(f"Unknown pruner: {pruner_method}")
        return pruner

    def objective(self, trial: optuna.Trial) -> float:

        kwargs = self._hyperparams.copy()

        trial.model_class = None
        if self.algo == "her":
            trial.model_class = self._hyperparams.get("model_class", None)

        # Hack to use DDPG/TD3 noise sampler
        trial.n_actions = self.n_actions
        # Sample candidate hyperparameters
        kwargs.update(HYPERPARAMS_SAMPLER[self.algo](trial))

        model = ALGOS[self.algo](
            env=self.create_envs(self.n_envs, no_log=True),
            tensorboard_log=None,
            # We do not seed the trial
            seed=None,
            verbose=0,
            **kwargs,
        )

        model.trial = trial

        eval_env = self.create_envs(n_envs=1, eval_env=True)

        eval_freq = int(self.n_timesteps / self.n_evaluations)
        # Account for parallel envs
        eval_freq_ = max(eval_freq // model.get_env().num_envs, 1)
        # Use non-deterministic eval for Atari
        eval_callback = TrialEvalCallback(
            eval_env,
            trial,
            n_eval_episodes=self.n_eval_episodes,
            eval_freq=eval_freq_,
            deterministic=self.deterministic_eval,
        )

        try:
            model.learn(self.n_timesteps, callback=eval_callback)
            # Free memory
            model.env.close()
            eval_env.close()
        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            # Free memory
            model.env.close()
            eval_env.close()
            # Prune hyperparams that generate NaNs
            print(e)
            raise optuna.exceptions.TrialPruned()
        is_pruned = eval_callback.is_pruned
        reward = eval_callback.last_mean_reward

        del model.env, eval_env
        del model

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return reward

    def hyperparameters_optimization(self) -> None:

        if self.verbose > 0:
            print("Optimizing hyperparameters")

        if self.storage is not None and self.study_name is None:
            warnings.warn(
                f"You passed a remote storage: {self.storage} but no `--study-name`."
                "The study name will be generated by Optuna, make sure to re-use the same study name "
                "when you want to do distributed hyperparameter optimization."
            )

        if self.tensorboard_log is not None:
            warnings.warn("Tensorboard log is deactivated when running hyperparameter optimization")
            self.tensorboard_log = None

        # TODO: eval each hyperparams several times to account for noisy evaluation
        sampler = self._create_sampler(self.sampler)
        pruner = self._create_pruner(self.pruner)

        if self.verbose > 0:
            print(f"Sampler: {self.sampler} - Pruner: {self.pruner}")

        study = optuna.create_study(
            sampler=sampler,
            pruner=pruner,
            storage=self.storage,
            study_name=self.study_name,
            load_if_exists=True,
            direction="maximize",
        )

        try:
            study.optimize(self.objective, n_trials=self.n_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            pass

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("Value: ", trial.value)

        print("Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        report_name = (
            f"report_{self.env_id}_{self.n_trials}-trials-{self.n_timesteps}"
            f"-{self.sampler}-{self.pruner}_{int(time.time())}.csv"
        )

        log_path = os.path.join(self.log_folder, self.algo, report_name)

        if self.verbose:
            print(f"Writing report to {log_path}")

        # Write report
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        study.trials_dataframe().to_csv(log_path)


