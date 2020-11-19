"""
train_on_env.py
train an agent on gym classic control environment.
"""

import argparse
import difflib
import importlib
import os
import uuid

############################
# set the python path properly
import sys
path_to_curr_file=os.path.realpath(__file__)
proj_root=os.path.dirname(os.path.dirname(path_to_curr_file))
if proj_root not in sys.path:
    sys.path.insert(0,proj_root)
############################
import time
import numpy as np
import seaborn
import torch as th
from stable_baselines3.common import logger
from stable_baselines3.common.utils import set_random_seed
import shutil
# Register custom envs
import experiment.envs.import_envs  # noqa: F401 pytype: disable=import-error
from experiment.utils.exp_manager import ExperimentManager
from experiment.utils.utils import title

seaborn.set()


CONFIGS_DIR = os.path.join(os.path.expanduser('~'),'share','Data','MLA','sbl3','configs')
LOGGER_NAME=os.path.splitext(os.path.basename(__file__))[0]

def parse_cmd_line():

    parser = argparse.ArgumentParser()
    parser.add_argument('exparams', type=str, help='experiment params file path')
    parser.add_argument('-d','--gpuid',type=str,default='',help='gpu id or "cpu"')
    parser.add_argument('--num_experiments', help='number of experiments', default=1,type=int)
    parser.add_argument('--seed', help='Random generator seed', type=int, default=1)
    parser.add_argument("--num-threads", help="Number of threads for PyTorch (-1 to use default)", default=-1, type=int)
    parser.add_argument('-i', '--trained_agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('-n', '--n_timesteps', help='Overwrite the number of timesteps', default=-1,type=int)
    parser.add_argument('--log_interval', help='Override log interval (default: -1, no change)', default=-1,type=int)
    args = parser.parse_args()
    return args


def run_experiment(experiment_params):
    '''
    parse the experiment params, create experiment manager and run the flow

    '''
    seed = experiment_params.seed
    logger.info(title(f"starting experiment seed {seed}", 30))

    # define the exp_manager
    exp_manager = ExperimentManager(experiment_params)

    add_log_formats=['csv']
    if experiment_params.log_tensorboard:
        add_log_formats.append('tensorboard')


    with logger.ScopedOutputConfig(exp_manager.output_dir,add_log_formats):
        # Prepare experiment and launch hyperparameter optimization if needed
        model = exp_manager.setup_experiment()

        # Normal training
        if model is not None:
            if experiment_params.n_timesteps>0:
                exp_manager.learn(model)
                exp_manager.save_trained_model(model)
            if experiment_params.expert_steps_to_record > 0:
                exp_manager.rollout_on_env(model)
        else:
            exp_manager.hyperparameters_optimization()

        logger.info(title("completed experiment seed {}".format(seed), 30))
    return




def main():

    args = parse_cmd_line()
    print('reading experiment params from '+args.exparams)
    exparams = os.path.splitext(args.exparams)[0]
    exparams = exparams[exparams.find('config')+7:]
    module_path = 'experiment.config.'+exparams.replace(os.path.sep,'.')
    exp_params_module = importlib.import_module(module_path)
    experiment_params = getattr(exp_params_module,'experiment_params')
    # set the path to the config file
    exparams_path = os.path.abspath(args.exparams)

    # create experiment folder and logger
    exp_folder_name = os.path.basename(exparams) + '-' + time.strftime("%d-%m-%Y_%H-%M-%S")
    experiment_params.output_root_dir = os.path.join(experiment_params.output_root_dir,exp_folder_name)
    os.makedirs(experiment_params.output_root_dir, exist_ok=True)
    # copy the configuration file
    shutil.copy(exparams_path,experiment_params.output_root_dir)

    logger.configure(os.path.join(experiment_params.output_root_dir), ['stdout', 'log'])

    # check if some cmd line arguments should override the experiment params
    if args.n_timesteps > -1:
        logger.info(f"overriding n_timesteps with {args.n_timesteps}")
        experiment_params.n_timesteps=args.n_timesteps
    if args.log_interval > -1:
        logger.info(f"overriding log_interval with {args.log_interval}")
        experiment_params.log_interval=args.log_interval

    logger.info(title(f'Starting {args.num_experiments} experiments', 40))
    for e in range(args.num_experiments):
        seed = args.seed+100*e
        experiment_params.seed = seed
        # run experiment will generate its own sub folder for each seed
        # not yet clear how to support hyper parameter search...
        set_random_seed(seed)
        if args.num_threads > 0:
            if args.verbose > 1:
                print(f"Setting torch.num_threads to {args.num_threads}")
            th.set_num_threads(args.num_threads)
        run_experiment(experiment_params)

    return

if __name__ == '__main__':
    main()
    sys.path.remove(proj_root)







