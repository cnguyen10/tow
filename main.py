import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms

from torcheval import metrics as torcheval_metrics

import aim

import numpy as np
import os
import argparse
import typing
import sys
import logging
import copy
import csv

from pathlib import Path

from scipy.optimize import minimize

from tqdm import tqdm

from Model import Model
# from TowLeastSquares import Tow
from EpisodeSampler import EpisodeSampler
from utils import train_val_split, vector_to_list_parameters, pil_loader_color, pil_loader_gray

# import subprocess
# from gcloud import download_file, upload_file

# --------------------------------------------------
# region SETUP INPUT PARSER
# --------------------------------------------------
parser = argparse.ArgumentParser(description='Setup variables')

parser.add_argument('--experiment-name', type=str, help='')
parser.add_argument('--description', default=None, help='Description of the run (store in aim.Run.description)')

parser.add_argument('--ds-folder', type=str, default='../datasets', help='Parent folder containing the dataset')
parser.add_argument('--datasource', action='append', help='List of datasets: omniglot, ImageNet')

parser.add_argument('--img-size', action='append', help='A pair of image size: 28 28 or 84 84')

parser.add_argument('--gray-img', dest='color_img', action='store_false')
parser.add_argument('--color-img', dest='color_img', action='store_true')
parser.set_defaults(color_img=True)

parser.add_argument('--ml-type', type=str, default='maml', help='maml or protonet')

parser.add_argument('--num-time-steps', type=int, default=10, help='Number of the finite horizon timesteps')

parser.add_argument('--first-order', dest='first_order', action='store_true')
parser.add_argument('--no-first-order', dest='first_order', action='store_false')
parser.set_defaults(first_order=True)

parser.add_argument('--network-architecture', type=str, default='CNN', help='The base model used, including CNN and ResNet18 defined in CommonModels')

# baselines
parser.add_argument('--baseline', action='append')

# Including learnable BatchNorm in the model or not learnable BN
parser.add_argument('--batchnorm', dest='batchnorm', action='store_true')
parser.add_argument('--no-batchnorm', dest='batchnorm', action='store_false')
parser.set_defaults(batchnorm=False)

# use strided convolution or max-pooling
parser.add_argument('--strided', dest='strided', action='store_true')
parser.add_argument('--no-strided', dest='strided', action='store_false')
parser.set_defaults(strided=True)

# set vectorization flag for Jacobian calculate
parser.add_argument('--vectorized', dest='vectorize', action='store_true')
parser.add_argument('--no-vectorized', dest='vectorize', action='store_false')
parser.set_defaults(vectorize=False)

parser.add_argument('--num-ways', type=int, default=5, help='Number of classes within an episode')

parser.add_argument('--num-inner-updates', type=int, default=5, help='The number of gradient updates for episode adaptation')
parser.add_argument('--inner-lr', type=float, default=0.1, help='Learning rate of episode adaptation step')

parser.add_argument('--logdir', type=str, default='.', help='Folder to store model and logs')

parser.add_argument('--meta-lr', type=float, default=3e-4, help='Learning rate for meta-update')
parser.add_argument('--minibatch', type=int, default=20, help='Minibatch of episodes to update meta-parameters')
parser.add_argument("--weight-decay", type=float, default=0, help="L2 regularization for meta-paramter")
parser.add_argument("--dropout-prob", type=float, default=0, help="Dropout probability")

parser.add_argument('--meta-lr-baseline', type=float, default=1e-3, help='Learning rate for the baselines, including uniform-weighting, exploration and exploitation')

parser.add_argument('--k-shot', type=int, default=1, help='Number of training examples per class')
parser.add_argument('--v-shot', type=int, default=15, help='Number of validation examples per class')
parser.add_argument('--k-shot-max', type=int, default=50, help='Number of maximum k-shot, applicable when k-shot = 0')

parser.add_argument('--jacobian-step', type=int, default=1, help='Number of steps when calculating Jacobian matrix for Gauss-Newton matrix')

# PRIOR
parser.add_argument('--beta-u', type=float, default=1, help='Precision prior for action (task weighting)')
parser.add_argument('--beta-x', type=float, default=0, help='Precision prior for state (model parameters)')

parser.add_argument('--mu-u', default=None, help='Prior mean for the action')

parser.add_argument('--mu0', type=float, default=0, help='Regularization to add to the diagonal of V')

parser.add_argument('--num-ilqr-iters', type=int, default=5, help='Number of iterations used in iLQR')

parser.add_argument('--num-episodes-per-epoch', type=int, default=10000, help='Save meta-parameters after this number of episodes')
parser.add_argument('--num-minibatches-print', type=int, default=10, help='Number of minibatches to add result to tensorboard')
parser.add_argument('--num-epochs', type=int, default=1, help='')
parser.add_argument('--resume-epoch', type=int, default=0, help='Resume')
parser.add_argument('--run-hash', default=None, help='Hash id of the run to resume')

parser.add_argument('--train', dest='train_flag', action='store_true')
parser.add_argument('--test', dest='train_flag', action='store_false')
parser.set_defaults(train_flag=True)

parser.add_argument('--tqdm', dest='tqdm', action='store_true')
parser.add_argument('--no-tqdm', dest='tqdm', action='store_false')
parser.set_defaults(tqdm=True)

parser.add_argument('--num-workers', type=int, default=0, help='Number of workers used in data-loader')

parser.add_argument('--num-testing-episodes', type=int, default=100, help='Number of episodes used in testing')

parser.add_argument('--dirichlet-concentration', type=float, default=1.1, help='Concentration of Dirichlet distribution used as prior for Exploration and Exploitation')

args = parser.parse_args()
# endregion

logging.basicConfig(level=logging.INFO)

config = {}
for key in args.__dict__:
    config[key] = args.__dict__[key]

# meta-learning type
if config['ml_type'] == 'maml':
    from Tow import Tow
else:
    from TowProto import Tow

logging.info('Dataset = {0}'.format(config['datasource']))

my_folder = os.path.join(
    config['ml_type'],
    ''.join(config['datasource']),
    config['network_architecture']
)

config['logdir'] = os.path.join(config['logdir'], my_folder)
if not os.path.exists(path=config['logdir']):
    Path(config['logdir']).mkdir(parents=True, exist_ok=True)
logging.info('Logdir = {0:s}'.format(config['logdir']))

# region CUDA
logging.info('CUDA is available: {0}'.format(torch.cuda.is_available()))
if not torch.cuda.is_available():
    raise ValueError('No CUDA')

# number of GPUs
logging.info('Number of GPU(s) = {0:d}'.format(torch.cuda.device_count()))

config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else torch.device('cpu'))

if (torch.cuda.device_count() >= 2):
    config['device2'] = torch.device('cuda:1')
else:
    config['device2'] = config['device']
# endregion

# PRIOR
if config['mu_u'] is None:
    config['mu_u'] = 1 / config['minibatch']
else:
    config['mu_u'] = float(config['mu_u'])


def get_data_one_minibatch(
    batch_size: int,
    eps_dataloader: typing.Iterator[torch.utils.data.DataLoader]
) -> typing.List[typing.Dict[str, torch.Tensor]]:
    """Get episodes and store their train-test split data

    Args:
        batch_size: the number of tasks within a mini-batch
        eps_loader: generates data of a task

    Return:
        A dictionary storing data of a mini-batch of tasks
            e.g. minibatch['y_t'].shape == (batch_size, k_shot)
    """
    minibatch = []

    task_count = 0
    while (task_count < batch_size):
        eps_data = next(eps_dataloader)

        # split data into train and validation
        if config['k_shot'] == 0:
            k_shot = np.random.randint(
                low=1,
                high=config['k_shot_max'] + 1, size=1
            ).item()
        else:
            k_shot = config['k_shot']
        split_data = train_val_split(
            eps_data=eps_data,
            k_shot=k_shot,
            v_shot=config['v_shot']
        )

        for key in split_data:
            split_data[key] = split_data[key].to(
                device=config['device'],
                non_blocking=True
            )

        minibatch.append(split_data)

        task_count += 1

    return minibatch


def generate_random_trajectory(
    x0: torch.Tensor,
    optimizer_sd: typing.Dict[str, typing.Any],
    uhats: typing.Union[typing.List[float], np.ndarray, torch.Tensor],
    minibatches: typing.List[typing.Dict[str, torch.Tensor]],
    tow: Tow
) -> typing.Tuple[torch.Tensor, typing.List[typing.Dict], torch.Tensor]:
    """
    Args:
        x0: initial paramter or initial state
        optimizer_sd: the initial state_dict of the optimizer used
        uhats: the action or weighting vector
        minibatches: data of tasks belonging to a minibatch

    Return:
        xhats: the nominal state corresponding to the nominal action uhats
        opt_sds: a list of optimizer's state dictionaries
        loss:
    """
    # initialize nominal state vector
    xhats = torch.empty(
        size=(len(minibatches), x0.numel()),
        device=config['device']
    )

    # initialize a list of optimizer's state dictionaries
    opt_sds = []

    # initialize uniform-weighted validation loss
    # loss = 0
    loss = torcheval_metrics.Mean(device=xhats.device)

    x_tp1 = x0
    optimizer_sd_temp = copy.deepcopy(optimizer_sd)
    for t in range(len(minibatches)):
        xhats[t, :] = x_tp1
        opt_sds.append(copy.deepcopy(optimizer_sd_temp))

        # reshape the model parameter vector to list of parameters
        params = vector_to_list_parameters(
            vec=xhats[t, :],
            parameter_shapes=tow.parameter_shapes
        )
        # enable gradient
        for param in params:
            param.requires_grad_()  # inline enable gradient

        # get new state
        new_params, loss_temp, optimizer_sd_temp = tow.train_one_minibatch_Adam(
            params=params,
            optimizer_sd=optimizer_sd_temp,
            minibatch=minibatches[t],
            eps_weight=uhats[t, :]
        )

        # loss += loss_temp
        loss.update(input=loss_temp)

        # flatten the parameters and remove gradient tracking
        with torch.no_grad():
            x_tp1 = torch.nn.utils.parameters_to_vector(parameters=new_params)

    return xhats, opt_sds, loss.compute()


if __name__ == "__main__":
    # create the task generator on the training set
    # define some transformation
    transformations_train = transforms.Compose(
        transforms=[
            transforms.Resize(size=([int(i) for i in config['img_size']])),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                transforms=[transforms.RandomRotation(degrees=(90, 90))],
                p=0.25
            ),
            transforms.RandomApply(
                transforms=[transforms.RandomRotation(degrees=(180, 180))],
                p=0.25
            ),
            transforms.RandomApply(
                transforms=[transforms.RandomRotation(degrees=(270, 270))],
                p=0.25
            ),
            transforms.ToTensor()
        ]
    )
    transformations_test = transforms.Compose(
        transforms=[
            transforms.Resize(size=([int(i) for i in config['img_size']])),
            transforms.ToTensor()
        ]
    )
    loader_fn = pil_loader_color if config['color_img'] else pil_loader_gray

    if config['train_flag']:
        train_dataset = torch.utils.data.ConcatDataset(
            datasets=[ImageFolder(
                root=os.path.join(config['ds_folder'], data_source, 'train'),
                transform=transformations_train,
                loader=loader_fn
            ) for data_source in config['datasource']]
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_sampler=EpisodeSampler(
                sampler=torch.utils.data.RandomSampler(
                    data_source=train_dataset
                ),
                num_ways=config['num_ways'],
                drop_last=True,
                num_samples_per_class=max(
                    config['k_shot'],
                    config['k_shot_max']
                ) + config['v_shot']
            ),
            num_workers=config['num_workers'],
            pin_memory=True
        )

    # create task generator on the testing set
    test_dataset = torch.utils.data.ConcatDataset(
        datasets=[ImageFolder(
            root=os.path.join(
                config['ds_folder'],
                data_source,
                'val' if config['train_flag'] else 'test'
            ),
            transform=transformations_test,
            loader=loader_fn
        ) for data_source in config['datasource']]
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_sampler=EpisodeSampler(
            sampler=torch.utils.data.RandomSampler(data_source=test_dataset),
            num_ways=config['num_ways'],
            drop_last=True,
            num_samples_per_class=max(
                config['k_shot'],
                config['k_shot_max']
            ) + config['v_shot']
        ),
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # turn on cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    # Initialize a meta-learning instance
    tow = Tow(config=config)
    tow.initialize_model(eps_dataloader=test_dataloader)

    # add tow to the list of testing algorithm
    if config['baseline'] is not None:
        config['baseline'].append('tow')
    else:
        config['baseline'] = tuple(('tow',))
    logging.info('Algorithms tested are: {}'.format(config['baseline']))

    # initialize a dictionary to store all models
    models = dict.fromkeys(config['baseline'])

    config['checkpoint_path'] = config['logdir']

    # region INITIALIZE / LOAD model
    if config['resume_epoch'] == 0:
        # initialze parametersd
        params_0 = []
        for param_shape in tow.parameter_shapes:
            if len(param_shape) >= 2:
                temp = torch.empty(size=param_shape)
                torch.nn.init.kaiming_normal_(tensor=temp, nonlinearity='relu')
            else:
                temp = torch.zeros(size=param_shape)
            params_0.append(temp)

        for key in models:
            config_temp = copy.deepcopy(x=config)
            # if key != 'tow':
            #     config_temp['meta_lr'] = config['meta_lr_baseline']

            models[key] = Model(
                params=params_0,
                requires_grad=True,
                config=config_temp,
                name=key
            )

            del config_temp

        del params_0
    else:
        if config['run_hash'] is not None:
            config['checkpoint_path'] = os.path.join(config['logdir'], config['run_hash'])

        saved_checkpoint = torch.load(
            f=os.path.join(
                config['checkpoint_path'],
                'Epoch_{0:d}.pt'.format(config['resume_epoch'])
            ),
            map_location=lambda storage, loc: storage.cuda(config['device'].index) if config['device'].type == 'cuda' else storage
        )

        models_temp = saved_checkpoint['models']

        for key in models_temp:
            models[key] = Model(
                params=models_temp[key].params,
                config=config,
                requires_grad=True,
                name=key,
                state_dict=models_temp[key].state_dict
            )
        del models_temp
    # endregion

    # for key in models:
    #     if key != 'tow':
    #         for i in range(len(models[key].state_dict['param_groups'])):
    #             models[key].state_dict['param_groups'][i]['lr'] /= 5

    # TRAIN
    if config['train_flag']:
        try:
            # initialize AIM repository
            aim_run = aim.Run(
                run_hash=config['run_hash'],
                repo='logs',
                read_only=False,
                experiment='{}'.format(args.experiment_name),
                # force_resume=False if config['run_hash'] is None else True,
                capture_terminal_logs=False,
                system_tracking_interval=1200
            )

            if args.description is not None:
                if aim_run.description is None:
                    aim_run.description = ''
                aim_run.description += '\n{:s}'.format(args.description)

            aim_run['hparams'] = {key: config[key] for key in config if isinstance(config[key], (int, bool, str, float))}

            if config['run_hash'] is None:
                config['checkpoint_path'] = os.path.join(config['logdir'], aim_run.hash)
                Path(config['checkpoint_path']).mkdir(parents=True, exist_ok=True)

            # initialize a nominal actions (task-weight vector)
            uhats_rand = torch.ones(
                size=(config['num_time_steps'], config['minibatch']),
                device=config['device']
            ) / config['minibatch']

            normal_u = torch.distributions.normal.Normal(
                loc=torch.tensor(config['mu_u'], device=config['device']),
                scale=torch.tensor(
                    data=np.sqrt(1 / config['beta_u']),
                    device=config['device']
                )
            )

            my_train_dataloader = iter(train_dataloader)
            my_test_dataloader = iter(test_dataloader)

            loss_prior_u = -torch.sum(
                input=normal_u.log_prob(value=uhats_rand)
            ).item()

            for epoch_id in tqdm(
                iterable=range(
                    config['resume_epoch'],
                    config['resume_epoch'] + config['num_epochs'],
                    1
                ),
                desc='Epoch',
                leave=False,
                position=0,
                disable=not config['tqdm']
            ):

                # variables to monitor losses
                loss_monitor = {key: torcheval_metrics.Mean(device=config['device']) for key in models}
                loss_temp = {key: 0. for key in models}

                for minibatch_id in tqdm(
                    iterable=range(
                        0,
                        config['num_episodes_per_epoch'] + 1,
                        config['num_time_steps']
                    ),
                    desc='Mini-batch index',
                    leave=False,
                    position=1,
                    disable=not config['tqdm']
                ):
                    # generate a number of mini-batches
                    minibatches = [get_data_one_minibatch(batch_size=config['minibatch'], eps_dataloader=my_train_dataloader) for _ in range(config['num_time_steps'])]

                    # minibatches_2 = copy.deepcopy(x=minibatches)

                    # for _ in range(5):
                    for minibatch in minibatches:
                    # region BASE-LINES
                    #     for minibatch_id_2, minibatch in enumerate(minibatches):
                    #         # augment task data
                    #         for task_id, task_data in enumerate(minibatch):
                    #             augment_flag = np.random.randint(low=0, high=4)
                    #             if augment_flag == 0:
                    #                 task_data['x_t'] = transforms.functional.gaussian_blur(img=minibatches_2[minibatch_id_2][task_id]['x_t'], kernel_size=(3, 3), sigma=(0.1, 0.1))

                        # region UNIFORM
                        # ---------------------------------------------------------------------------------
                        if 'uniform' in models:
                            models['uniform'].params, loss_temp['uniform'], models['uniform'].state_dict = tow.train_one_minibatch_Adam(
                                params=models['uniform'].params,
                                optimizer_sd=models['uniform'].state_dict,
                                minibatch=minibatch,
                                eps_weight=[1 / config['minibatch']] * config['minibatch']
                            )

                            if torch.isnan(input=loss_temp['uniform']):
                                raise ValueError('Training for uniform-weighting: loss is NaN')

                            # loss_monitor['uniform'].append(loss_temp['uniform'])
                            loss_monitor['uniform'].update(loss_temp['uniform'])
                        # ---------------------------------------------------------------------------------
                        # endregion

                        # region EXPLORATION
                        # ---------------------------------------------------------------------------------
                        # calculate validation losses of a mini-batch
                        if 'exploration' in models:
                            with torch.no_grad():
                                params_temp = torch.nn.utils.parameters_to_vector(parameters=models['exploration'].params)
                            params_temp.requires_grad_(requires_grad=True)

                            val_losses = tow.validation_loss(
                                param_vec=params_temp,
                                minibatch=minibatch,
                                eps_weight=[1.] * config['minibatch']
                            )
                            f_temp = lambda u: np.sum(-u * val_losses.cpu().detach().numpy() - (config['dirichlet_concentration'] - 1) * np.log(u))
                            df_temp = lambda u: -val_losses.cpu().detach().numpy() - (config['dirichlet_concentration'] - 1) / u

                            # def f_temp(u: np.ndarray) -> float:
                            #     return np.sum(-u * val_losses.cpu().detach().numpy() - (config['dirichlet_concentration'] - 1) * np.log(u))

                            # def df_temp(u: np.ndarray) -> float:
                            #     return -val_losses.cpu().detach().numpy() - (config['dirichlet_concentration'] - 1) / u

                            ans = minimize(
                                fun=f_temp,
                                x0=np.ones(shape=(config['minibatch'],)) / config['minibatch'],
                                method='SLSQP',
                                jac=df_temp,
                                bounds=([(1e-6, 1)] * config['minibatch']),
                                constraints=({'type': 'eq', 'fun': lambda u: np.sum(u) - 1})
                            )

                            models['exploration'].params, loss_temp['exploration'], models['exploration'].state_dict = tow.train_one_minibatch_Adam(
                                params=models['exploration'].params,
                                optimizer_sd=models['exploration'].state_dict,
                                minibatch=minibatch,
                                eps_weight=ans.x
                            )

                            if torch.isnan(input=loss_temp['exploration']):
                                raise ValueError("Training for exploration: loss is NaN")

                            # loss_monitor['exploration'].append(loss_temp['exploration'])
                            loss_monitor['exploration'].update(loss_temp['exploration'])
                        # ---------------------------------------------------------------------------------
                        # endregion

                        # region EXPLOITATION
                        # ---------------------------------------------------------------------------------
                        if 'exploitation' in models:
                            with torch.no_grad():
                                params_temp = torch.nn.utils.parameters_to_vector(parameters=models['exploitation'].params)
                            params_temp.requires_grad_(requires_grad=True)

                            val_losses = tow.validation_loss(
                                param_vec=params_temp,
                                minibatch=minibatch,
                                eps_weight=[1.] * config['minibatch']
                            )
                            f_temp = lambda u: np.sum(u * val_losses.cpu().detach().numpy() - (config['dirichlet_concentration'] - 1) * np.log(u))
                            df_temp = lambda u: val_losses.cpu().detach().numpy() - (config['dirichlet_concentration'] - 1) / u

                            # def f_temp(u: np.ndarray) -> float:
                            #     return np.sum(u * val_losses.cpu().detach().numpy() - (config['dirichlet_concentration'] - 1) * np.log(u))

                            # def df_temp(u: np.ndarray) -> float:
                            #     val_losses.cpu().detach().numpy() - (config['dirichlet_concentration'] - 1) / u

                            ans = minimize(
                                fun=f_temp,
                                x0=np.ones(shape=(config['minibatch'],)) / config['minibatch'],
                                method='SLSQP',
                                jac=df_temp,
                                bounds=([(1e-6, 1)] * config['minibatch']),
                                constraints=({'type': 'eq', 'fun': lambda u: np.sum(u) - 1})
                            )

                            models['exploitation'].params, loss_temp['exploitation'], models['exploitation'].state_dict = tow.train_one_minibatch_Adam(
                                params=models['exploitation'].params,
                                optimizer_sd=models['exploitation'].state_dict,
                                minibatch=minibatch,
                                eps_weight=ans.x
                            )

                            if torch.isnan(input=loss_temp['exploitation']):
                                raise ValueError('Training for exploitation: loss is NaN')

                            # loss_monitor['exploitation'].append(loss_temp['exploitation'])
                            loss_monitor['exploitation'].update(loss_temp['exploitation'])
                        # ---------------------------------------------------------------------------------
                        # endregion

                    # endregion
                    # minibatches = copy.deepcopy(x=minibatches_2)

                    # region TOW
                    # ---------------------------------------------------------------------------------
                    with torch.no_grad():
                        xhat_0 = torch.nn.utils.parameters_to_vector(
                            parameters=models['tow'].params
                        )
                    for param in models['tow'].params:
                        param.requires_grad_(requires_grad=False)

                    # generate random trajectory
                    xhats_rand, opt_sds, loss_rand = generate_random_trajectory(
                        x0=xhat_0,
                        optimizer_sd=models['tow'].state_dict,
                        uhats=uhats_rand,
                        minibatches=minibatches,
                        tow=tow
                    )

                    # iLQR to find good trajectory
                    xhats, uhats, opt_sd, losses_tow_temp, x_tp1 = tow.iLQR(
                        xhats=xhats_rand,
                        opt_sds=opt_sds,
                        uhats=uhats_rand,
                        minibatches=minibatches,
                        num_iters=config['num_ilqr_iters'],
                        total_cost_hat=loss_rand + loss_prior_u
                    )

                    # check if there is error causing NaN
                    if (torch.isnan(input=losses_tow_temp).any()):
                        raise ValueError("Training for weighted: loss is NaN.")

                    # add loss to a list to monitor
                    for i in range(len(minibatches)):
                        # loss_monitor['tow'].append(losses_tow_temp[i])
                        loss_monitor['tow'].update(losses_tow_temp[i])

                    # update TOW model
                    models['tow'].params = vector_to_list_parameters(
                        vec=x_tp1,
                        parameter_shapes=tow.parameter_shapes
                    )
                    models['tow'].state_dict = copy.deepcopy(opt_sd)
                    # endregion

                    if (minibatch_id % config['num_minibatches_print'] == 0) and (minibatch_id != 0):
                        # calculate the global step
                        global_step = minibatch_id + epoch_id * config['num_episodes_per_epoch']

                        for key in loss_monitor:
                            aim_run.track(
                                value=loss_monitor[key].compute(),
                                name='Loss',
                                epoch=epoch_id,
                                step=global_step,
                                context={'model': key, 'subset': 'train'}
                            )

                        # # reset monitoring variables
                        loss_monitor = {key: torcheval_metrics.Mean(device=config['device']) for key in models}

                        # region TEST
                        # -----------------------------------------------------------------------------
                        # enable gradient
                        for param in models['tow'].params:
                            param.requires_grad_()

                        test_minibatch = get_data_one_minibatch(
                            batch_size=config['num_testing_episodes'],
                            eps_dataloader=my_test_dataloader
                        )

                        # initialize some variables for evaluation
                        accuracies = dict.fromkeys([key for key in models.keys()])
                        NLLs = dict.fromkeys([key for key in models.keys()])

                        # enable evaluation mode to disable dropout
                        tow.f_base_net.eval()

                        for key in models.keys():
                            accuracies[key], NLLs[key] = tow.evaluate_one_minibatch(params=models[key].params, minibatch=test_minibatch)

                        # enable training mode
                        tow.f_base_net.train()

                        for key in accuracies:
                            aim_run.track(
                                name='Accuracy',
                                value=np.mean(a=accuracies[key]),
                                step=global_step,
                                epoch=epoch_id,
                                context={'subset': 'test', 'model': key}
                            )

                            aim_run.track(
                                name='Accuracy_std',
                                value=np.std(a=accuracies[key]),
                                step=global_step,
                                epoch=epoch_id,
                                context={'subset': 'test', 'model': key}
                            )

                        for key in NLLs:
                            aim_run.track(
                                name='Loss',
                                value=np.mean(a=NLLs[key]),
                                step=global_step,
                                epoch=epoch_id,
                                context={'model': key, 'subset': 'test'}
                            )

                        accuracies = None
                        NLLs = None
                        # -----------------------------------------------------------------------------
                        # endregion
                # save
                with torch.no_grad():
                    checkpoint = {'models': models}

                checkpoint_path = os.path.join(config['checkpoint_path'], 'Epoch_{0:d}.pt'.format(epoch_id + 1))
                torch.save(obj=checkpoint, f=checkpoint_path)
                # logging.info(msg='Parameters are saved into {0:s}\n'.format(checkpoint_path))
        finally:
            aim_run.close()
            logging.info(msg='\nProgram is terminated.')

    else:  # TEST

        # enable EVALUATION mode
        for key in models.keys():
            # enable evaludation mode to disable dropout
            tow.f_base_net.eval()

        # enable gradient
        for param in models['tow'].params:
            param.requires_grad_()

        accuracies = dict.fromkeys([key for key in models.keys()])

        try:
            out_csv = open(file=os.path.join(config['logdir'], 'test.csv'), mode='w', newline='')
            csv_writer = csv.writer(out_csv, delimiter=',', quoting=csv.QUOTE_NONE, escapechar='\n')

            header = ['#']
            header.extend(list(models.keys()))

            csv_writer.writerow(header)

            for i, eps_data in enumerate(test_dataloader):
                # get data of a task
                if config['k_shot'] == 0:
                    k_shot = np.random.randint(low=1, high=config['k_shot_max'] + 1, size=1).item()
                else:
                    k_shot = config['k_shot']
                eps_split_data = train_val_split(eps_data=eps_data, k_shot=k_shot, v_shot=config['v_shot'])
                for key in eps_split_data:
                    eps_split_data[key] = eps_split_data[key].to(config['device'])

                row_result = []
                # evaluate
                for key in models.keys():
                    accuracies[key], _ = tow.evaluate_one_minibatch(params=models[key].params, minibatch=[eps_split_data])
                    row_result.append(accuracies[key][0])

                # write to output file
                csv_writer.writerow(row_result)

                sys.stdout.write('\x1b[1A\x1b[2K')
                print(i)

                if i >= config['num_testing_episodes']:
                    break
        finally:
            out_csv.close()
            print('Close csv file')
