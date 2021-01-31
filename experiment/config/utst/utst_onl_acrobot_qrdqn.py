from experiment.config.default_config import *

##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'acrobot'

#################
# Policy        #
#################
policy = 'MlpPolicy'
policy_kwargs={'net_arch': [256,256],'n_quantiles': 25}

##########################################################
# Agent Params                                           #
# copied from the qrdqn.yml
##########################################################
agent_params = QRDQNAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = policy
agent_params.learning_rate = 6.3e-4
agent_params.batch_size = 128
agent_params.buffer_size = 50000
agent_params.learning_starts = 0
agent_params.gamma = 0.99
agent_params.target_update_interval = 250
agent_params.train_freq = 4
agent_params.gradient_steps = -1
agent_params.exploration_fraction = 0.12
agent_params.exploration_final_eps= 0.1
agent_params.policy_kwargs = policy_kwargs



##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 100000
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.expert_steps_to_record = 0  # number of episodes to record into the experience buffer
experiment_params.online_eval_freq = int(experiment_params.n_timesteps/20)  # evaluate on eval env every this number of timesteps
experiment_params.online_eval_n_episodes = 30
experiment_params.name = __name__.split('.')[-1]





