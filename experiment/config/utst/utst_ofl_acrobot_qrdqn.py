from experiment.config.default_config import *


##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'acrobot'

##########################################################
# Experience Buffer or Expert generator
##########################################################
experience_dataset = '/home/gkoren2/share/Data/MLA/sbl3/results/utst_onl_acrobot_rnd-16-12-2020_12-52-46/1/er_Acrobot-v1_random_100000.npz'
# experience_dataset = '/home/gkoren2/share/Data/MLA/sbl3/results/utst_onl_acrobot_dqn-17-12-2020_15-30-58/1/er_Acrobot-v1_dqn_100000.npz'

#################
# Policy        #
#################
policy = 'MlpPolicy'



##########################################################
# Agent Params                                           #
##########################################################
agent_params = QRDQNAgentParams()
# here we can change the various parameters - for example, we can change the batch size
agent_params.policy = policy
agent_params.learning_rate = 1e-4
agent_params.batch_size = 128
agent_params.gamma = 0.99
agent_params.target_update_interval = 500     # in offline mode, measure by minibatches
agent_params.tau = 1.0                      # perform hard update (copy the parameters)
agent_params.gradient_steps = 1
agent_params.policy_kwargs = dict(net_arch=[64],n_quantiles=50)


##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.env_params = env_params
experiment_params.expert_data = experience_dataset
experiment_params.n_timesteps = 1000000
experiment_params.agent_params = agent_params
experiment_params.expert_steps_to_record = 50000  # number of episodes to record into the experience buffer
experiment_params.online_eval_freq = int(experiment_params.n_timesteps/50)  # evaluate on eval env every this number of timesteps
experiment_params.online_eval_n_episodes = 30
experiment_params.name = __name__.split('.')[-1]




