from experiment.config.default_config import *

##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'lunland'

#################
# Policy        #
#################
policy = 'MlpPolicy'
policy_kwargs={'net_arch': [256,256],'n_quantiles': 170}




##########################################################
# Agent Params                                           #
# copied from the qrdqn.yml
##########################################################
agent_params = QRDQNAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = policy
agent_params.learning_rate = 'lin_1.5e-3'
agent_params.batch_size = 128
agent_params.buffer_size = 100000
agent_params.learning_starts = 10000
agent_params.gamma = 0.995
agent_params.target_update_interval = 1
agent_params.train_freq = 256
agent_params.gradient_steps = -1
agent_params.exploration_fraction = 0.24
agent_params.exploration_final_eps= 0.18
agent_params.policy_kwargs = policy_kwargs



##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 1e5
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.expert_steps_to_record = 0  # number of episodes to record into the experience buffer
experiment_params.online_eval_freq = int(experiment_params.n_timesteps/100)  # evaluate on eval env every this number of timesteps
experiment_params.online_eval_n_episodes = 30
experiment_params.name = __name__.split('.')[-1]





