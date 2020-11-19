from experiment.config.default_config import *

##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'mntcar'
env_params.norm_reward = True
env_params.norm_obs = True

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
agent_params.n_steps = 16
agent_params.gae_lambda = 0.98
agent_params.gamma = 0.99
agent_params.n_epochs = 4
agent_params.ent_coef = 0.0



##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 1e6
experiment_params.n_envs = 16
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.expert_steps_to_record = 50000  # number of episodes to record into the experience buffer
experiment_params.online_eval_freq = int(experiment_params.n_timesteps/100)  # evaluate on eval env every this number of timesteps
experiment_params.online_eval_n_episodes = 30
experiment_params.name = __name__.split('.')[-1]





