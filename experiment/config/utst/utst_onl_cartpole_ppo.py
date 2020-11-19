from experiment.config.default_config import *

##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'cartpole'

#################
# Policy        #
#################
policy = 'MlpPolicy'


##########################################################
# Agent Params                                           #
##########################################################
agent_params = PPOAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = policy

agent_params.learning_rate = 'lin_0.001'
agent_params.clip_range = 'lin_0.2'
agent_params.n_steps = 32
agent_params.batch_size = 256
agent_params.gae_lambda = 0.8
agent_params.gamma = 0.98
agent_params.n_epochs = 20
agent_params.ent_coef = 0.0

##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 2e5
experiment_params.n_envs = 8
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.expert_steps_to_record = 50000  # number of episodes to record into the experience buffer
experiment_params.online_eval_freq = int(experiment_params.n_timesteps/100)  # evaluate on eval env every this number of timesteps
experiment_params.online_eval_n_episodes = 30
experiment_params.name = __name__.split('.')[-1]





