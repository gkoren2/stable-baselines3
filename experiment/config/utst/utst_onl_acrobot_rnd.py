from experiment.config.default_config import *


##########################################################
# Env                                                    #
##########################################################
env_params = EnvParams()
env_params.env_id = 'acrobot'

#################
# Policy        #
#################


##########################################################
# Agent Params                                           #
##########################################################
agent_params = RandomAgentParams()
# here we can change the various parameters - for example, we can change the batch size


##########################################################
# Experiment                                             #
##########################################################
experiment_params = ExperimentParams()
experiment_params.n_timesteps = 0
experiment_params.env_params = env_params
experiment_params.agent_params = agent_params
experiment_params.expert_steps_to_record = 100000  # number of episodes to record into the experience buffer
experiment_params.name = __name__.split('.')[-1]





