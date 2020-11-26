# utst_onl_acrobot_dqn_ro.py
# config file to load pretrained model and perform rollout, without continue training
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


##########################################################
# Agent Params                                           #
##########################################################
agent_params = DQNAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = policy
agent_params.learning_rate = 6.3e-4
agent_params.exploration_final_eps= 0.1
agent_params.exploration_fraction = 0.12

agent_params.buffer_size = 50000
agent_params.batch_size = 128
agent_params.learning_starts = 0
agent_params.gamma = 0.99
agent_params.target_update_interval = 250
agent_params.gradient_steps = -1
# agent_params.policy_kwargs = dict(net_arch=[256, 256])


##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.trained_agent_model_file='/home/gkoren2/share/Data/MLA/sbl3/results/utst_onl_acrobot_dqn-26-11-2020_12-29-12/1/best_model.zip'
experiment_params.n_timesteps = 0
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params

experiment_params.expert_steps_to_record = 100000  # number of episodes to record into the experience buffer
experiment_params.online_eval_freq = int(experiment_params.n_timesteps/10)  # evaluate on eval env every this number of timesteps
experiment_params.online_eval_n_episodes = 30
experiment_params.name = __name__.split('.')[-1]





