from experiment.config.default_config import *

##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'mntcar'

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
agent_params.learning_rate = 4e-3
agent_params.batch_size = 128
agent_params.buffer_size = 10000
agent_params.learning_starts = 1000
agent_params.gamma = 0.98
agent_params.target_update_interval = 600
agent_params.train_freq = 16
agent_params.gradient_steps = 8
agent_params.exploration_fraction = 0.2
agent_params.exploration_final_eps= 0.07
agent_params.policy_kwargs=dict(net_arch=[256, 256])



##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.trained_agent_model_file='/home/gkoren2/share/Data/MLA/sbl3/results/utst_onl_mntcar_dqn-24-11-2020_22-04-04/1/final_model.zip'
experiment_params.n_timesteps = 1.2e5
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.expert_steps_to_record = 50000  # number of episodes to record into the experience buffer
experiment_params.online_eval_freq = int(experiment_params.n_timesteps/100)  # evaluate on eval env every this number of timesteps
experiment_params.online_eval_n_episodes = 30
experiment_params.name = __name__.split('.')[-1]





