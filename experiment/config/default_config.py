import os
import logging
import numpy as np
from experiment.envs.dtt_env import BENCHMARKS,PLATFORMS
from experiment.utils import ALGOS
ALGO_IDS = list(ALGOS.keys())
from experiment.envs.import_envs import CC_ENVS
#############################
# Env Defaults

class EnvParams:
    def __init__(self,env_id='cartpole'):
        self.env_id=CC_ENVS.get(env_id,env_id)
        # self.normalize = False      # can also be "{'norm_obs': True, 'norm_reward': False}"
        # consider dropping normalize and use the below directly
        self.norm_obs = False
        self.norm_reward = False

        self.env_wrapper = None     # see utils.wrappers
        self.frame_stack = 1        # 1 = no stack , >1 means how many frames to stack
        self.env_kwargs = {}

    def as_dict(self):
        return vars(self)



class DTTRealCSVParams(EnvParams):
    def __init__(self):
        super(DTTRealCSVParams, self).__init__()
        self.env_id = 'DTTRealCSV'
        self.obs_dim = None         # dictated by the experience buffer




class DTTEnvSimParams(EnvParams):
    def __init__(self):
        super(DTTEnvSimParams, self).__init__()
        self.env_id = 'DTTSim'
        self.episode_workloads = 5*(['cb15']+[('cooldown',1)]) + [('cooldown',600)] + 5*(['cb20']+[('cooldown',1)])

        # note: to create 10 iterations of the above for each episode, do :
        # self.workload = 10*([BENCHMARKS['cb15']] + [BENCHMARKS['cooldown']] * 150)
        self.platform = PLATFORMS['Scarlet']
        self.log_output = None
        self.full_reset = True
        self.use_wrapper = False
        self.wrapper_params = {'feature_extractor':None,'reward_calc':None,'n_frames':5,'norm_params_file':None}

#############################
# Agents Defaults

class AgentParams:
    def __init__(self):

        # these are parameters that should be derived from the experiment params
        self.verbose = 1
        self.tensorboard_log = None
        self.create_eval_env = False
        self._init_setup_model = True
        self.seed = None
        self.policy_kwargs = None
        self.device = 'auto'
        return

    def as_dict(self):
        return vars(self)

class RandomAgentParams(AgentParams):
    def __init__(self):
        super(RandomAgentParams, self).__init__()
        self.algorithm = 'random'
        return

class DQNAgentParams(AgentParams):
    """
    Parameters for DQN agent
    The agent gets the following values in its construction:
    policy,env
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    """
    def __init__(self):
        super(DQNAgentParams, self).__init__()
        # Default parameters for DQN Agent
        self.algorithm = 'dqn'

        self.policy = 'MlpPolicy'    # or 'CnnPolicy' or 'CustomDQNPolicy'

        self.learning_rate = 1e-4
        self.buffer_size= 1000000
        self.learning_starts = 50000
        self.batch_size = 32
        self.tau = 1.0
        self.gamma = 0.99
        self.train_freq = 4
        self.gradient_steps = 1
        self.n_episodes_rollout = -1
        self.optimize_memory_usage = False
        self.target_update_interval = 10000
        self.exploration_fraction = 0.1
        self.exploration_initial_eps = 1.0
        self.exploration_final_eps = 0.05
        self.max_grad_norm = 10

        # offline RL parameters
        self.buffer_train_fraction = 1.0    # 1.0 = use all for train. evaluation done on env
                                            # 0.8 = use 0.8 for train, ope on rest 0.2


        return


class QRDQNAgentParams(AgentParams):
    """
    Parameters for QRDQN agent
    Quantile Regression Deep Q-Network (QR-DQN)
    Paper: https://arxiv.org/abs/1710.10044
    Default hyperparameters are taken from the paper and are tuned for Atari games.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping (if None, no clipping)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(self):
        super(QRDQNAgentParams, self).__init__()
        # Default parameters for DQN Agent
        self.algorithm = 'qrdqn'

        self.policy = 'MlpPolicy'    # or 'CnnPolicy'

        self.learning_rate = 5e-5
        self.buffer_size= 1000000
        self.learning_starts = 50000
        self.batch_size = 32
        self.tau = 1.0
        self.gamma = 0.99
        self.train_freq = 4
        self.gradient_steps = 1
        self.n_episodes_rollout = -1
        self.optimize_memory_usage = False
        self.target_update_interval = 10000
        self.exploration_fraction = 0.1
        self.exploration_initial_eps = 1.0
        self.exploration_final_eps = 0.05
        self.max_grad_norm = None

        # offline RL parameters
        self.buffer_train_fraction = 1.0    # 1.0 = use all for train. evaluation done on env
                                            # 0.8 = use 0.8 for train, ope on rest 0.2


        return


class PPOAgentParams(AgentParams):
    """
    PPOAgentParams
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.

    """
    def __init__(self):
        super(PPOAgentParams, self).__init__()
        self.algorithm = 'ppo'

        self.policy = 'MlpPolicy'

        self.learning_rate = 3e-4
        self.n_steps = 2048
        self.batch_size = 64
        self.n_epochs = 10
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        # self.clip_range_vf = None
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.use_sde = False
        self.sde_sample_freq = -1
        self.target_kl = None
        return


class SACAgentParams(AgentParams):
    """
    SACAgentParams:
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)

    """
    def __init__(self):
        super(SACAgentParams, self).__init__()
        # parameters to be parsed and removed at runtime
        self.algorithm='sac'
        self.policy = 'MlpPolicy'       # see also 'CustomSACPolicy'
        self.learning_rate = 3e-4       # can also be 'lin_3e-4'
        self.buffer_size = int(1e6)

        self.learning_starts = 100
        self.batch_size = 256
        self.tau = 0.005
        self.gamma = 0.99
        self.train_freq = 1
        self.gradient_steps = 1
        self.n_episodes_rollout = -1
        self.action_noise = None
        self.optimize_memory_usage = False
        self.ent_coef = "auto"
        self.target_update_interval = 1
        self.target_entropy = "auto"
        self.use_sde = False
        self.sde_sample_freq = -1
        self.use_sde_at_warmup = False
        return


class DDPGAgentParams(AgentParams):
    """
    DDPGAgentParams
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    """
    def __init__(self):
        super(DDPGAgentParams, self).__init__()
        self.algorithm='ddpg'
        # parameters to be parsed at runtime
        # self.noise_type = 'ornstein-uhlenbeck'
        # self.noise_std = 0.5

        self.policy = 'MlpPolicy'
        self.learning_rate = 1e-3
        self.buffer_size = int(1e6)
        self.learning_starts = 100
        self.batch_size = 100
        self.tau = 0.001
        self.gamma = 0.99
        self.train_freq = -1
        self.gradient_steps = -1
        self.n_episodes_rollout = 1
        self.action_noise = None
        self.optimize_memory_usage = False
        return


class A2CAgentParams(AgentParams):
    """
    A2CAgentParams
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rms_prop_eps: RMSProp epsilon. It stabilizes square root computation in denominator
        of RMSProp update
    :param use_rms_prop: Whether to use RMSprop (default) or Adam as optimizer
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param normalize_advantage: Whether to normalize or not the advantage
    """
    def __init__(self):
        super(A2CAgentParams, self).__init__()
        self.algorithm='a2c'

        self.policy = 'MlpPolicy'

        self.learning_rate = 7e-4
        self.n_steps = 5
        self.gamma = 0.99
        self.gae_lambda = 1.0
        self.ent_coef = 0.0
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.rms_prop_eps = 1e-5
        self.use_rms_prop = True
        self.use_sde = False
        self.sde_sample_freq = -1
        self.normalize_advantage = False
        return


class TD3AgentParams(AgentParams):
    """
    TD3AgentParams
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    """
    def __init__(self):
        super(TD3AgentParams, self).__init__()
        # parameters to be parsed at runtime and removed
        self.algorithm='td3'
        # self.noise_type = 'ornstein-uhlenbeck'
        # self.noise_std = 0.5

        self.learning_rate = 1e-3
        self.buffer_size = int(1e6)
        self.learning_starts = 100
        self.batch_size = 100
        self.tau = 0.005
        self.gamma = 0.99
        self.train_freq = -1
        self.gradient_steps = -1
        self.n_episodes_rollout = 1
        self.action_noise = None
        self.optimize_memory_usage = False
        self.policy_delay = 2
        self.target_policy_noise = 0.2
        self.target_noise_clip = 0.5
        return


class DBCQAgentParams(AgentParams):
    """
    Parameters for DQN agent
    The agent gets the following values in its construction:
    policy,env
    gamma = 0.99, learning_rate = 5e-4, buffer_size = 50000, exploration_fraction = 0.1,
    exploration_final_eps = 0.02, exploration_initial_eps = 1.0, train_freq = 1, batch_size = 32, double_q = True,
    learning_starts = 1000, target_network_update_freq = 500, prioritized_replay = False,
    prioritized_replay_alpha = 0.6, prioritized_replay_beta0 = 0.4, prioritized_replay_beta_iters = None,
    prioritized_replay_eps = 1e-6, param_noise = False,

    n_cpu_tf_sess = None, verbose = 0, tensorboard_log = None, _init_setup_model = True, policy_kwargs = None,
    full_tensorboard_log = False, seed = None
    """
    def __init__(self):
        super(DBCQAgentParams, self).__init__()
        # Default parameters for DQN Agent
        self.algorithm = 'dbcq'
        self.policy = 'MlpPolicy'    # or 'CnnPolicy' or 'CustomDQNPolicy' - the main policy that we train
        self.learning_rate = 1e-4               # can also be 'lin_<float>' e.g. 'lin_0.001'
        self.target_network_update_freq = 1   # number of epochs between target network updates
        self.param_noise = False
        self.act_distance_thresh = 0.3          # if gen_act_policy is Neural Net - corresponds to the threshold tau
                                                # i.e. actions with likelihood ratio larger than threshold will be
                                                # considered as candidates
                                                # if gen_act_policy is KNN - the max distance from nearest neighbor
                                                # s.t. actions that are farther will be thrown
        # other default params
        self.gamma = 0.99
        self.batch_size = 32
        self.buffer_train_fraction = 1.0        # 100% will be used for training the policy and the reward model for DM
                                                # the rest (20%) will be used for Off policy evaluation
        # parameters of the generative model for actions
        self.gen_act_policy = None               # 'KNN' for K nearest neighbors, 'NN' for Neural Net
                                                # if 'NN' the agent will use the same type of policy for the generative model
        self.gen_act_params = {'type': 'NN', 'n_epochs': 50, 'lr': 1e-3, 'train_frac': 0.7, 'batch_size': 64}
        # self.gen_act_params = {'type':'KNN','size': 1000}  # knn parameters
        self.gen_train_with_main = False        # if True, continue to train the generative model while training the
                                                # main agent
        self.n_cpu_tf_sess = None
        self.policy_kwargs = None
        return

#################################
# Experiment Params
class ExperimentParams:
    def __init__(self):
        self.name=None      # should be overriden
        self.seed = 1
        ####### Folders #######
        self.output_root_dir = os.path.join(os.path.expanduser('~'),'share','Data','MLA','sbl3','results')

        ####### Logging #######
        self.log_level = logging.INFO
        # use the %(process)d format to log the process-ID (useful in multiprocessing where each process has a different ID)
        # LOG_FORMAT = '%(asctime)s | %(process)d | %(message)s'
        self.log_format = '%(asctime)s | %(message)s'
        self.log_date_format = '%y-%m-%d %H:%M:%S'
        self.log_tensorboard = True
        self.verbose = 1
        self.log_interval = -1      # -1 to use agent-specific defaults (see learn method)
        ####### Env #######
        self.n_envs = 1
        self.env_params = None


        ############### Agent ################
        # pretrain with behavioral cloning
        # (pretrain_dataset,pretrain_expert_agent) are related as follows:
        # dataset = None, expert_agent = None : no pretraing
        # dataset != None, expert_agent = None : pretrain from existing buffer (behavioral cloning)
        # dataset = None, expert_agent != None : Illegal option. agent must have path to save
        # dataset != None, expert_agent != None : use expert to write to pretrain_dataset and then pretrain


        # we can either load a trained model and continue train it using the agent_params


        # given we do pretraining, use the following for pretrain (behavioral cloning)
        self.expert_data = None          # path to experience buffer that can be wrapped by ExpertData
                                                # if None, there's no pre-train
        # imitation learning parameters
        self.pretrain_params = {'n_epochs':0,'lr':1e-4,'train_frac':0.8,'batch_size':64}

        self.save_replay_buffer = False         # whether to save the replay buffer used for training
                                                # valid only for off-policy algorithms

        ########################
        # trained agent - if we want to continue training from a saved agent
        self.trained_agent_model_file=None         # path to main pretrained agent - to continue training
        self.agent_params = None        # agent that trains the main policy.
                                        # should be class of agent params e.g. DQNAgentParams()
        # training params
        self.n_timesteps = 1e5          # number of timesteps to train main policy
        self.log_interval = -1          # using algorithm default

        self.online_eval_freq = 0        # evaluate on eval env and/or with ope every this number of timesteps
        self.online_eval_n_episodes = 10

        self.off_policy_eval_dataset_eval_fraction = 0      # fraction of the data that will be used for evaluation
                                                            # the rest will be used for training the agent and the reward model

        self.checkpoint_save_freq = -1              # Save the model every n steps (if negative, no checkpoint)

        self.truncate_last_trajectory = True    # When using HER with online sampling the last trajectory in the
                                                # replay buffer will be truncated after reloading the replay buffer.

        self.expert_steps_to_record = 0        # number of episodes to record into the experience buffer






        ###### Hyper Parameters Optimization ######
        self.hpopt_on = False           # whether to perform hyperparameters optimization
        self.hpopt_storage = None       # Database storage path if distributed optimization should be used
        self.hpopt_n_trials = 10          # maximum number of trials for finding the best hyperparams
        self.hpopt_n_jobs = 1           # number of parallel jobs when doing hyperparameter search
        self.hpopt_sampler = 'tpe'      # Sampler to use when optimizing hyperparameters (["random", "tpe", "skopt"])
        self.hpopt_pruner = 'median'    # Pruner to use when optimizing hyperparameters (["halving", "median", "none"])
        self.hpopt_n_startup_trials = 10      # Number of trials before using optuna sampler
        self.hpopt_n_evaluations = 20         # Number of evaluations for hyperparameter optimization
        self.hpopt_study_name = None          # Study name for distributed optimization

    def as_dict(self):
        return vars(self)





