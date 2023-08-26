import torch
import higher

from torcheval import metrics as torcheval_metrics

import numpy as np
import typing
import copy
import logging

from CommonModels import CNN, ResNet18, MiniCNN, ResNet10
from utils import train_val_split, vector_to_list_parameters


class Tow(object):
    def __init__(self, config: dict) -> None:
        """Initialize an instance of MAML model based on the input configuration
        """
        self.config = config

        self.num_total_samples = self.config['v_shot'] * self.config['num_ways']
        self.incr = min(self.config['jacobian_step'], self.num_total_samples)

        # initialize a normal distribution to calculate loss for action u
        self.normal_u = torch.distributions.normal.Normal(
            loc=torch.tensor(config['mu_u'], device=config['device']),
            scale=torch.tensor(np.sqrt(1 / config['beta_u']), device=config['device'])
        )

    def initialize_model(self, eps_dataloader: torch.utils.data.DataLoader) -> None:
        """Initialize or load model by adding functional/stateless base module and its parameter shapes
        """

        if self.config['network_architecture'] == 'CNN':
            base_net = CNN(
                dim_output=self.config['num_ways'],
                bn_affine=self.config['batchnorm'],
                stride_flag=self.config['strided']
            )
        elif (self.config["network_architecture"] == "ResNet10"):
            base_net = ResNet10(
                dim_output=self.config['num_ways'],
                bn_affine=self.config['batchnorm'],
                dropout_prob=self.config["dropout_prob"]
            )
        elif self.config['network_architecture'] == 'ResNet18':
            base_net = ResNet18(
                dim_output=self.config['num_ways'],
                bn_affine=self.config['batchnorm'],
                dropout_prob=self.config["dropout_prob"]
            )
        elif self.config['network_architecture'] == 'MiniCNN':
            base_net = MiniCNN(dim_output=self.config['num_ways'], bn_affine=self.config['batchnorm'])
        else:
            raise NotImplementedError('Network architecture is unknown. Please implement it in the CommonModels.py.')

        # ---------------------------------------------------------------
        # run a dummy task to initialize lazy modules defined in base_net
        # ---------------------------------------------------------------
        for eps_data in eps_dataloader:
            # split data into train and validation
            if self.config['k_shot'] == 0:
                k_shot = np.random.randint(low=1, high=self.config['k_shot_max'] + 1, size=1).item()
            else:
                k_shot = self.config['k_shot']
            eps_split_data = train_val_split(eps_data=eps_data, k_shot=k_shot, v_shot=self.config['v_shot'])

            logging.info('Image size = {}'.format(tuple(eps_split_data['x_t'][0].shape)))

            # run to initialize lazy modules
            base_net.forward(eps_split_data['x_t'])
            break

        params = torch.nn.utils.parameters_to_vector(parameters=base_net.parameters())
        self.num_params = params.numel()
        logging.info('Number of parameters of the base network = {0:,}.'.format(self.num_params))

        # # move to xpu
        # base_net.to(self.config['device'])

        # add functional_base_net into self
        self.f_base_net = higher.patch.make_functional(module=base_net)
        self.f_base_net.track_higher_grads = False
        self.f_base_net._fast_params = [[]]

        # add running_mean and running_var for BatchNorm2d
        for m in self.f_base_net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.running_mean = None
                m.running_var = None

        # add shapes of parameters into self
        self.parameter_shapes = []
        for param in base_net.state_dict().values():
            self.parameter_shapes.append(param.shape)

        return None

    def adapt_to_episode(self, x: torch.Tensor, y: torch.Tensor, params: typing.List[torch.Tensor]) -> torch.Tensor:
        """
        """
        # initialize the initial parameters for the base model
        q_params = tuple(p + 0. for p in params)

        for _ in range(self.config['num_inner_updates']):
            y_logits = self.f_base_net.forward(x, params=q_params)
            cls_loss = torch.nn.functional.cross_entropy(input=y_logits, target=y)

            grads = torch.autograd.grad(
                outputs=cls_loss,
                inputs=q_params,
                retain_graph=True,
                create_graph=not self.config['first_order']
            )

            new_q_params = []
            for param, grad in zip(q_params, grads):
                new_q_params.append(higher.optim._add(tensor=param, a1=-self.config['inner_lr'], a2=grad))

            q_params = tuple(new_q_param + 0. for new_q_param in new_q_params)

        return q_params

    def evaluate_one_minibatch(self, params: typing.List[torch.Tensor], minibatch: typing.List[typing.Dict[str, torch.Tensor]]) -> typing.Tuple[typing.List[float], typing.List[float]]:
        """Evaluate the performance
        """
        accuracies = [None] * len(minibatch)
        val_losses = [None] * len(minibatch)

        # device = params[0].device

        for i, task_data in enumerate(minibatch):
            # move data to GPU (if there is a GPU)
            x_t = task_data['x_t']
            y_t = task_data['y_t']
            x_v = task_data['x_v']
            y_v = task_data['y_v']

            # adaptation using the support data
            q_params = self.adapt_to_episode(x=x_t, y=y_t, params=params)

            # predition
            y_pred = self.f_base_net.forward(x_v, params=q_params)

            accuracies[i] = (y_pred.argmax(dim=1) == y_v).float().mean().item()
            val_losses[i] = torch.nn.functional.cross_entropy(input=y_pred, target=y_v, reduction='mean').item()

            # sys.stdout.write('\033[F')
            # print(i + 1)

        # acc_mean = np.mean(a=accuracies)
        # acc_std = np.std(a=accuracies)
        # print('\nAccuracy = {0:.2f} +/- {1:.2f}\n'.format(acc_mean * 100, 1.96 * acc_std / np.sqrt(len(accuracies)) * 100))
        return accuracies, val_losses

    def train_one_minibatch(
        self,
        params: typing.List[torch.Tensor],
        minibatch: typing.List[typing.Dict[str, torch.Tensor]],
        eps_weight: typing.Union[np.ndarray, torch.Tensor, typing.List[float]],
        weight_decay: float = 0
    ) -> torch.Tensor:
        """
        """
        loss_monitor = 0.

        opt = torch.optim.SGD(params=params, lr=self.config['meta_lr'], weight_decay=weight_decay)
        opt.zero_grad(set_to_none=True)

        # device = params[0].device

        for i, task_data in enumerate(minibatch):
            # get training and validation subsets of each task
            x_t = task_data['x_t']
            y_t = task_data['y_t']
            x_v = task_data['x_v']
            y_v = task_data['y_v']

            # adaptation using the support data
            q_params = self.adapt_to_episode(x=x_t, y=y_t, params=params)

            # predition
            y_logits = self.f_base_net.forward(x_v, params=q_params)

            cls_loss = torch.nn.functional.cross_entropy(input=y_logits, target=y_v, reduction='mean')
            cls_loss = cls_loss * eps_weight[i]

            cls_loss.backward()

            loss_monitor += cls_loss.item() / (eps_weight[i] * len(minibatch))

        opt.step()
        opt.zero_grad(set_to_none=True)

        return params, loss_monitor

    def train_one_minibatch_Adam(
        self,
        params: typing.List[torch.Tensor],
        optimizer_sd: dict,
        minibatch: typing.List[typing.Dict[str, torch.Tensor]],
        eps_weight: typing.Union[np.ndarray, torch.Tensor, typing.List[float]]
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        """
        # loss_monitor = 0.
        loss_monitor = torcheval_metrics.Mean(device=params[0].device)

        optimizer = torch.optim.Adam(params=params, lr=self.config['meta_lr'])
        optimizer.load_state_dict(state_dict=optimizer_sd)
        optimizer.zero_grad(set_to_none=True)

        # device = params[0].device

        for i, task_data in enumerate(minibatch):
            # get training and validation subsets of each task
            x_t = task_data['x_t']
            y_t = task_data['y_t']
            x_v = task_data['x_v']
            y_v = task_data['y_v']

            # adaptation using the support data
            q_params = self.adapt_to_episode(x=x_t, y=y_t, params=params)

            # predition
            y_logits = self.f_base_net.forward(x_v, params=q_params)

            cls_loss = torch.nn.functional.cross_entropy(input=y_logits, target=y_v, reduction='mean')
            weighted_cls_loss = cls_loss * eps_weight[i]

            weighted_cls_loss.backward()

            # loss_monitor += cls_loss.item()
            loss_monitor.update(cls_loss)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # update state_dict of the optimizer
        new_optimizer_sd = optimizer.state_dict()

        # loss_monitor /= len(minibatch)

        return params, loss_monitor.compute(), new_optimizer_sd

    def validation_loss(self, param_vec: torch.Tensor, minibatch: typing.List[typing.Dict[str, torch.Tensor]], eps_weight: typing.Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Calculate the validation loss (mainly used to calculate the transition dynamics and cost)

        Args:
            param_vec: vector of the flatten model meta-parameters
            param_id: the index of the parameter group that has its gradient tracked
            minibatch: the minibatch of tasks
            eps_weight: the weighting contribution of each task in the minibatch to the learning of the meta-parameter

        Return: average loss on validation subsets
        """
        # initialize variable to store validation losses
        val_loss = torch.empty(len(minibatch), dtype=torch.float, device=param_vec.device)

        # reshape the parameter vector to list of tensors
        params = vector_to_list_parameters(vec=param_vec, parameter_shapes=self.parameter_shapes)

        for i, task_data in enumerate(minibatch):
            # get training and validation subsets of each task
            x_t = task_data['x_t']
            y_t = task_data['y_t']
            x_v = task_data['x_v']
            y_v = task_data['y_v']

            # adaptation using the support data
            q_params = self.adapt_to_episode(x=x_t, y=y_t, params=params)

            # predition
            y_logits = self.f_base_net.forward(x_v, params=q_params)

            cls_loss = torch.nn.functional.cross_entropy(input=y_logits, target=y_v, reduction='mean')
            cls_loss = cls_loss * eps_weight[i]

            val_loss[i] = cls_loss

        return val_loss

    def validation_loss_one_task(self, param_vec: torch.Tensor, eps_split_data: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
        # reshape the parameter vector to list of tensors
        params = vector_to_list_parameters(vec=param_vec, parameter_shapes=self.parameter_shapes)

        # get training and validation subsets of each task
        x_t = eps_split_data['x_t']
        y_t = eps_split_data['y_t']
        x_v = eps_split_data['x_v']
        y_v = eps_split_data['y_v']

        # adaptation using the support data
        q_params = self.adapt_to_episode(x=x_t, y=y_t, params=params)

        # predition
        y_logits = self.f_base_net.forward(x_v, params=q_params)

        cls_loss = torch.nn.functional.cross_entropy(input=y_logits, target=y_v, reduction='mean')

        return cls_loss

    def prediction_one_task(self, param_vec: torch.Tensor, eps_split_data: typing.Dict[str, torch.Tensor]) -> torch.Tensor:
        """Prediction of data within a task
        """
        # reshape the parameter vector to list of tensors
        params = vector_to_list_parameters(vec=param_vec, parameter_shapes=self.parameter_shapes)

        # get training and validation subsets of each task
        x_t = eps_split_data['x_t']
        y_t = eps_split_data['y_t']
        x_v = eps_split_data['x_v']

        # adaptation using the support data
        q_params = self.adapt_to_episode(x=x_t, y=y_t, params=params)

        return self.f_base_net.forward(x_v, params=q_params)  # (v_shot * num_classes, num_classes)

    # ---------------------------------------------------------------------------------------------
    # LQR-related functions
    # ---------------------------------------------------------------------------------------------
    def get_hessian_sigma_matrix(
        self,
        x: torch.Tensor,
        task_data: typing.Dict[str, torch.Tensor],
        epsilon: float = 1e-6
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """calculate the Hessian matrix of the softmax output w.r.t. the pre-softmax values

        Args:
            x: parameter of the neural network of interest
            task_data: data points of a task
            epsilon: extra value added to the diagonal of the Hessian to prevent underflow

        Returns:
            hessian: the Hessian matrix of interest
            sigma_f: the pre-softmax values
        """
        # enable gradient to perform task adaptation/prediction
        x.requires_grad_(True)
        f_pre_activation = self.prediction_one_task(param_vec=x, eps_split_data=task_data)  # (num_samples, num_classes)
        x.requires_grad_(False)
        f_pre_activation = f_pre_activation.detach()

        # activation values
        sigma_f = torch.nn.functional.softmax(input=f_pre_activation, dim=-1)  # (num_samples, num_classes)

        # calculate Hessian of the loss w.r.t. f_pre_activation
        hessian_sigma = - sigma_f[:, :, None] * sigma_f[:, None, :]  # (num_samples, num_classes, num_classes)
        # diagonal_temp = torch.diagonal(input=hessian_sigma, dim1=-2, dim2=-1) # (num_samples, num_classes)
        hessian_sigma[:, range(self.config['num_ways']), range(self.config['num_ways'])] += sigma_f + epsilon

        return hessian_sigma, sigma_f

    def gauss_newton_components(self, x: torch.Tensor, task_data: typing.Dict[str, torch.Tensor]) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """calculate the components made up Gauss_Newton_matrix with softmax activation
            G = J.T @ Hsigma @ J

        Args:
            x: parameter of the neural network of interest
            task_data: data points of a task

        Returns:
            jacobian_f: Jacobian matrix of loss w.r.t. x without activation function
            hessian_sigma: the Hessian matrix of activation values w.r.t. pre-activation values
            sigma_f: the activation values of the network
        """
        # region PARTIAL PARALLEL
        # allocate Jacobian matrix of pre-activation values w.r.t. x
        jacobian_f = torch.empty(
            size=(self.num_total_samples, self.config['num_ways'], self.num_params),
            device=self.config['device']
        )  # (num_total_samples, num_classes, X)

        pointer = 0
        eps_split_data = {
            'x_t': task_data['x_t'],
            'y_t': task_data['y_t']
        }
        # for _ in range(0, self.config['v_shot'], 1):
        while (pointer < self.num_total_samples):
            pointer_next = min(pointer + self.incr, self.num_total_samples)

            eps_split_data['x_v'] = task_data['x_v'][pointer:pointer_next]
            eps_split_data['y_v'] = task_data['y_v'][pointer:pointer_next]

            jacobian_f[pointer:pointer_next, :, :] = torch.autograd.functional.jacobian(
                func=lambda x: self.prediction_one_task(param_vec=x, eps_split_data=eps_split_data),
                inputs=x,
                create_graph=False,
                vectorize=self.config['vectorize']
            )

            pointer = pointer_next
        # endregion

        # # region PARALLEL
        # jacobian_f = torch.autograd.functional.jacobian(
        #     func=lambda x: self.prediction_one_task(param_vec=x, eps_split_data=task_data),
        #     inputs=x,
        #     create_graph=False,
        #     vectorize=self.config['vectorize']
        # )
        # # endregion

        # # region SEQUENTIAL
        # jacobian_f = torch.autograd.functional.jacobian(
        #     func=lambda x: self.prediction_one_task(param_vec=x, eps_split_data=task_data),
        #     inputs=x,
        #     create_graph=False,
        #     vectorize=self.config['vectorize']
        # ) # (num_total_samples, num_classes, X)
        # # endregion

        # # enable gradient to perform task adaptation/prediction
        # x.requires_grad_(True)
        # f_pre_activation = self.prediction_one_task(param_vec=x, eps_split_data=task_data) # (num_samples, num_classes)
        # x.requires_grad_(False)
        # f_pre_activation = f_pre_activation.detach()

        # # activation values
        # sigma_f = torch.nn.functional.softmax(input=f_pre_activation, dim=-1) # (num_samples, num_classes)

        # # calculate Hessian of the loss w.r.t. f_pre_activation
        # hessian_sigma = - sigma_f[:, :, None] * sigma_f[:, None, :] # (num_samples, num_classes, num_classes)
        # # diagonal_temp = torch.diagonal(input=hessian_sigma, dim1=-2, dim2=-1) # (num_samples, num_classes)
        # hessian_sigma[:, range(self.config['num_ways']), range(self.config['num_ways'])] += sigma_f + 1e-6

        hessian_sigma, sigma_f = self.get_hessian_sigma_matrix(
            x=x,
            task_data=task_data,
            epsilon=1e-6
        )

        return jacobian_f, hessian_sigma, sigma_f

    def jacobian_gauss_newton_matrices(self, xhat_t: torch.Tensor, minibatch: typing.List[typing.Dict[str, torch.Tensor]]) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """calculate Jacobian matrices and the diagonal of Gauss-Newton matrices for a minibatch of tasks

        Args:
            xhat_t: the nominal state/parameter
            minibatch: a list of task data

        Returns:
            grad_matrix: each column is the gradient of each task w.r.t. x
            gauss_newton_diagonals: a tensor/list of Gauss-Newton diagonals
        """
        # initialize Jacobian matrix of the loss on each task w.r.t. model parameter
        # list of gradients of tasks within a minibatch of tasks
        grad_matrix = torch.empty(
            size=(self.config['minibatch'], self.num_params),
            device=self.config['device']
        )  # (U, X)

        # initialize the diagonal of Gauss-Newton matrix
        gauss_newton_diagonals = torch.empty(
            size=(self.config['minibatch'], self.num_params),
            device=self.config['device']
        )  # (U, X)

        for task_id, task_data in enumerate(minibatch):  # for each task in minibatch t-th
            jacobian_f, hessian_sigma, sm_outputs = self.gauss_newton_components(
                x=xhat_t,
                task_data=task_data
            )  # jacobian_f = (num_samples, C, X)

            # add epsilon to diagonal to prevent error
            L_sigma, _ = torch.linalg.cholesky_ex(input=hessian_sigma)  # (num_samples, C, C)

            # getting labels
            y_v = task_data['y_v']
            # convert int labels into one-hot encoding
            y_v = torch.nn.functional.one_hot(
                input=y_v,
                num_classes=self.config['num_ways']
            )  # (num_samples, num_classes)

            # Jacobian vector for one task
            dy_v = sm_outputs - y_v  # (num_samples, num_classes)
            grad_matrix[task_id, :] = torch.sum(input=dy_v[:, :, None] * jacobian_f, dim=(0, 1)) / (self.config['v_shot'] * self.config['num_ways'])  # (X, )

            # calculate the factorization of Gauss-Newton matrix for each data-point
            # G = B.T @ B
            # gauss_newton_factorized = B
            gauss_newton_factorized = torch.transpose(
                input=L_sigma,
                dim0=-2,
                dim1=-1
            ) @ jacobian_f  # (num_samples, C, X)

            # set jacobian_f to none to save GPU memory
            jacobian_f = None

            # diagonal of Gauss-Newton matrix for each data-point
            gauss_newton_diagonal = torch.sum(
                input=torch.square(input=gauss_newton_factorized),
                dim=1
            )  # (num_samples, X)

            # reset gauss_newton_factorized to save GPU memory
            gauss_newton_factorized = None

            # diagonal of Gauss-Newton matrix for a task
            gauss_newton_diagonals[task_id, :] = torch.mean(
                input=gauss_newton_diagonal,
                dim=0
            )  # (X, )

            # reset to none to save GPU memory
            gauss_newton_diagonal = None

        return grad_matrix, gauss_newton_diagonals

    def get_dynamics_cost_adam(
        self,
        xhat_t: torch.Tensor,
        optimizer_sd: dict,
        uhat_t: torch.Tensor,
        minibatch: typing.List[typing.Dict[str, torch.Tensor]]
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate the Taylor's approximation of the state transition dynamics and cost function

        Args:
            xhat_t: the nominal state
            optimizer_sd: the state dictionary of Adam optimizer of the nominal state
            uhat_t: the nominal action
            minibatch: data of tasks within a mini-batch

        Returns:
            Fx: the diagonal of the weighted validation loss (Taylor first order of Adam)
            Fu: gradient of the state-transition dynamics w.r.t. action u
            Cxx: the diagonal of Gauss-Newton matrix of uniformly-weighted validation loss (cost) w.r.t. x
            Cuu: Hessian matrix of cost function w.r.t. u
            cx: gradient of cost function w.r.t. x
            cu: gradient of cost function w.r.t. u
        """
        # # Jacobian matrix of uniformly-weighted loss for tasks within a mini-batch
        # jacobian = torch.empty(size=(self.config['minibatch'], self.num_params), device=self.config['device'])
        # for task_id, task_data in enumerate(minibatch):
        #     jacobian[task_id, :] = torch.autograd.functional.jacobian(
        #         func=lambda x: self.validation_loss_one_task(param_vec=x, eps_split_data=task_data),
        #         inputs=xhat_t,
        #         create_graph=False,
        #         strict=False,
        #         vectorize=self.config['vectorize']
        #     ) # size (X)

        # Jacobian and Gauss-Newton matrices
        # - jacobian = (X, U)
        # - gauss-newton-diagonals = (U, X)
        jacobian, gauss_newton_diagonals = self.jacobian_gauss_newton_matrices(
            xhat_t=xhat_t,
            minibatch=minibatch
        )

        # region COST
        # Cxx is a diagonal with size (X, )
        Cxx = torch.sum(input=gauss_newton_diagonals, dim=0, keepdim=False)  # (X,)

        # Cuu is a diagonal matrix proportional to identity matrix
        Cuu = self.config['beta_u'] * torch.eye(
            n=self.config['minibatch'],
            device=self.config['device']
        )

        cx = torch.sum(input=jacobian, dim=0, keepdim=False)  # (X,)
        cu = self.config['beta_u'] * (uhat_t - self.config['mu_u'])  # (U,)
        # endregion

        # region DYNAMICS
        # region ADAM
        # get time step index and exponential moving averages at time step t - 1
        time_index = 1
        m_t_1 = torch.zeros(size=(self.num_params, 1), device=self.config['device'])
        v_t_1 = torch.zeros(size=(self.num_params, 1), device=self.config['device'])
        pointer = 0
        if len(optimizer_sd['state']) > 0:
            time_index = optimizer_sd['state'][0]['step'] + 1

            for param_id in range(len(optimizer_sd['state'])):
                num_param = np.prod(a=self.parameter_shapes[param_id])

                m_t_1[pointer:(pointer + num_param)] = torch.flatten(
                    input=optimizer_sd['state'][param_id]['exp_avg']
                )[:, None]
                v_t_1[pointer:(pointer + num_param)] = torch.flatten(
                    input=optimizer_sd['state'][param_id]['exp_avg_sq']
                )[:, None]

                pointer += num_param

        # some constants
        beta1 = optimizer_sd['param_groups'][0]['betas'][0]
        beta2 = optimizer_sd['param_groups'][0]['betas'][1]
        epsilon = optimizer_sd['param_groups'][0]['eps']

        d_weighted_loss_dx = jacobian.T @ uhat_t[:, None]  # (X, 1)

        # running averages of Adam optimizer
        m_t = beta1 * m_t_1 + (1 - beta1) * d_weighted_loss_dx
        v_t = beta2 * v_t_1 + (1 - beta2) * torch.square(input=d_weighted_loss_dx)
        # endregion

        den = epsilon + torch.sqrt(v_t / (1 - beta2 ** time_index))  # an intermediate variable to calculate dF

        # gradients w.r.t. state x
        dmdx = (1 - beta1) * gauss_newton_diagonals.T @ uhat_t[:, None]  # diagonal (X, 1) -> real size = (X, X)
        dvdx = 2 * (1 - beta2) * (jacobian.T @ uhat_t[:, None]) * (gauss_newton_diagonals.T @ uhat_t[:, None])  # diagonal (X, 1)

        # calculate Fx
        dFdx = dmdx / den - m_t / torch.square(input=den) * dvdx / (2 * torch.sqrt(input=(1 - beta2 ** time_index) * v_t))
        Fx = 1 - self.config['meta_lr'] / (1 - beta1 ** time_index) * dFdx  # (X, 1)
        Fx = torch.squeeze(input=Fx)  # (diagonal with size = (X, ))

        dFdx = None

        # gradients w.r.t. action u
        dmdu = (1 - beta1) * jacobian.T  # (X, U)
        dvdu = 2 * (1 - beta2) * jacobian.T * d_weighted_loss_dx  # (X, U)

        # calculate Fu
        dFdu = dmdu / den - m_t / torch.square(input=den) * dvdu / (2 * torch.sqrt(input=(1 - beta2 ** time_index) * v_t))  # (X, U)
        Fu = -self.config['meta_lr'] / (1 - beta1 ** time_index) * dFdu  # (X, U)

        # endregion

        # # reset to save GPU memory
        # dmdu = None
        # dvdu = None
        # jacobian = None

        # # initialize Hessian-matrix product with shape (X, U)
        # Cxx_Fu = torch.zeros(size=(self.num_params, self.config['minibatch']), device=self.config['device'])
        # H_Fu_weighted = torch.zeros(size=(self.num_params, self.config['minibatch']), device=self.config['device'])

        # for task_id, task_data in enumerate(minibatch):
        #     for u_id in range(self.config['minibatch']):
        #         Fu_H_temp = torch.autograd.functional.vhp(
        #             func=lambda x: self.validation_loss_one_task(param_vec=x, eps_split_data=task_data),
        #             inputs=xhat_t,
        #             v=Fu[:, u_id],
        #             create_graph=False,
        #             strict=False
        #         )[-1] # shape = (X)

        #         # accumulate to FuT_H
        #         Cxx_Fu[:, u_id] += Fu_H_temp
        #         H_Fu_weighted[:, u_id] += uhat_t[task_id] * Fu_H_temp

        # # reset variable to save GPU memory
        # Fu_H_temp = None

        # dmdx_Fu = (1 - beta1) * H_Fu_weighted # (X, U)
        # dvdx_Fu = 2 * (1 - beta2) * d_weighted_loss_dx * H_Fu_weighted # (X, U)

        # Fx_Fu = dmdx_Fu / den
        # Fx_Fu -= m_t / torch.square(input=den) * dvdx_Fu / (2 * torch.sqrt(input=(1 - beta2 ** time_index) * v_t))
        # Fx_Fu *= -self.config['meta_lr'] / (1 - beta1**time_index)
        # Fx_Fu += Fu

        # m_t = None
        # v_t = None
        # dmdx_Fu = None
        # dvdx_Fu = None

        return Fx, Fu, Cxx, Cuu, cx[:, None], cu[:, None]

    def LQR_backward(
        self,
        xhats: torch.Tensor,
        opt_sds: typing.List[dict],
        uhats: typing.Union[np.ndarray, torch.Tensor],
        minibatches: typing.List[typing.Dict[str, torch.Tensor]]
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        """Run the backward recursion for LQR model
        Args:
            xhats, uhats: random trajectory to calculate the cost and dynamics

        Return: linear controller: matrix K and vector k
        """
        # Initialize controller gain
        K = torch.empty(
            size=(len(minibatches), uhats.shape[1], xhats.shape[1]),
            device=self.config['device']
        )
        k = torch.empty(
            size=(len(minibatches), uhats.shape[1], 1),
            device=self.config['device']
        )

        # Initialize intermediate variables V and v
        # FuT_V = torch.zeros(size=(self.config['minibatch'], self.num_params), device=self.config['device']) # (U, X)
        V = torch.zeros(size=(self.num_params,), device=self.config['device'])
        v = torch.zeros(size=(self.num_params, 1), device=self.config['device'])

        for t in range(self.config['num_time_steps'] - 1, -1, -1):  # backward in time
            Fx, Fu, Cxx, Cuu, cx, cu = self.get_dynamics_cost_adam(
                xhat_t=xhats[t, :],
                optimizer_sd=opt_sds[t],
                uhat_t=uhats[t, :],
                minibatch=minibatches[t]
            )

            Qxx = Cxx + torch.square(input=Fx) * V

            Qxu = (Fx[:, None] * V[:, None]) * Fu

            Quu = Cuu + (Fu.T * V[None, :]) @ Fu

            qu = cu + Fu.T @ v
            qx = cx + Fx[:, None] * v

            # linear controller
            # Quu_inv = torch.linalg.inv(input=Quu)
            # K[t, :, :] = - Quu_inv @ Qxu.T
            # k[t, :, :] = - Quu_inv @ qu
            K[t, :, :] = - torch.linalg.solve(A=Quu, B=Qxu.T)
            k[t, :, :] = - torch.linalg.solve(A=Quu, B=qu)

            if (t == 0):
                return K, k

            # update vector v and diagonal matrix V
            V = Qxx + torch.sum(input=Qxu * K[t, :, :].T, dim=1, keepdim=False)
            v = qx + Qxu @ k[t, :, :]

        #     # calculate pseudo-inverse using conjugate gradient
        #     FuT_inv = CG_right(A=Fu.T, max_iterations=self.config['num_cg'], tol=1e-6) # (X, U)

        #     Qux = FuT_V @ Fx_Fu @ FuT_inv.T # = Fu(t).T @ V(t + 1) @ Fx(t) with resulting shape (U, X)
        #     Quu = Cuu + FuT_V @ Fu # (U, U)

        #     qu = cu + Fu.T @ v # (U, 1)

        #     qx = Fx_Fu.T @ v # (U, 1)
        #     qx = cx + FuT_inv @ qx # (X, 1)

        #     # linear controller
        #     Quu_inv = torch.linalg.inv(input=Quu)
        #     K[t, :, :] = - Quu_inv @ Qux
        #     k[t, :, :] = - Quu_inv @ qu

        #     if (t == 0):
        #         return K, k

        #     # update vector v
        #     v = qx + Qux.T @ k[t, :, :]

        #     # region CALCULATE Fu(t-1).T @ V(t), named as FuT_V_prev

        #     # calculate matrices and vectors of cost and dynamics at previous time step
        #     Fx_Fu_prev, Fu_prev, Cxx_Fu_prev, Cuu_prev, cx_prev, cu_prev = self.get_dynamics_cost_adam(
        #         xhat_t=xhats[t - 1, :],
        #         optimizer_sd=opt_sds[t - 1],
        #         uhat_t=uhats[t - 1, :],
        #         minibatch=minibatches[t - 1]
        #     )

        #     # 1st term of Fu(t-1).T @ V(t) is: Fu(t-1).T @ Cxx(t)
        #     FuT_V_prev = Fu_prev.T @ Cxx_Fu @ FuT_inv.T # (U, X)

        #     # 2nd term is: Fu(t-1).T @ Fx(t).T @ V(t) @ Fx
        #     temp = Fu_prev.T @ FuT_inv.T @ Fx_Fu.T # = Fu(t-1).T @ Fx.T with resulting shape (U, X)
        #     temp @= FuT_inv # = Fu(t-1).T @ Fx.T @ Fu.T.inv with resulting shape (U, U)
        #     temp @= FuT_V # = Fu(t-1).T @ Fx(t).T @ V(t) with resulting shape (U, X)
        #     temp @= Fx_Fu # (U, U)
        #     temp @= FuT_inv.T # (U, X)
        #     FuT_V_prev += temp # accumulate the 2nd term

        #     # 3rd term:
        #     temp = Fu_prev.T @ Qux.T # (U, U)
        #     temp @= K[t, :, :] # (U, X)
        #     FuT_V_prev += temp
        #     # endregion

        #     # region UPDATE dynamics and cost to make previous become the current
        #     FuT_V = FuT_V_prev
        #     Fx_Fu = Fx_Fu_prev
        #     Fu = Fu_prev
        #     Cxx_Fu = Cxx_Fu_prev
        #     Cuu = Cuu_prev
        #     cx = cx_prev
        #     cu = cu_prev
        #     # endregion

        #     # region RESET previous variables to save GPU memory
        #     FuT_V_prev = None
        #     Fx_Fu_prev = None
        #     Fu_prev = None
        #     Cxx_Fu_prev = None
        #     Cuu_prev = None
        #     cx_prev = None
        #     cu_prev = None
        #     temp = None
        #     # endregion

        return None, None

    def LQR_forward(
        self,
        xhats: torch.Tensor,
        optimizer_sd: dict,
        uhats: torch.Tensor,
        K: torch.Tensor,
        k: torch.Tensor,
        minibatches: typing.List[typing.Dict[str, torch.Tensor]],
        alpha: float
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.List[dict], torch.Tensor, torch.Tensor]:
        """Forward pass to calculate new trajectory

        Args:
            xhats: the nominal state
            optimizer_sd: the "initial" state dictionary of Adam optimizer
            uhats: the nominal action
            K: controller gain matrix
            k: controller open-loop vector
            minibatches: data of tasks in a minibatch
            alpha: backtracking line search coefficient

        Return:
            x: new state
            u: new action
            opt_sds: a list of Adam optimizer's state dictionaries corresponding to the new state and action
            losses: a list of losses for monitoring purpose
            x_tp1: x(T + 1)
        """
        # initalize variables for new trajectory
        x = torch.empty_like(input=xhats, device=self.config['device'])
        u = torch.empty_like(input=uhats, device=self.config['device'])

        # initialize a list placeholder to store Adam optimizer's state dictionary
        opt_sds = []

        # initialize a list to store losses for monitoring purpose
        # losses = np.empty(shape=(len(minibatches),))
        losses = torch.empty(size=(len(minibatches),), device=x.device)

        # set first state
        x_tp1 = xhats[0, :]

        # copy optimizer_state_dict
        optimizer_sd_temp = copy.deepcopy(optimizer_sd)

        for t in range(len(minibatches)):
            x[t, :] = x_tp1
            dx = x[t, :] - xhats[t, :]

            # add a copy of state dict into the list of state dicts
            opt_sds.append(copy.deepcopy(optimizer_sd_temp))

            u_t = K[t, :, :] @ dx[:, None] + alpha * k[t, :, :] + uhats[t, :][:, None]
            u[t, :] = torch.squeeze(input=u_t)

            # get new state from the dynamics
            # reshape the model parameter vector to list of parameters
            # params = vector_to_list_parameters(vec=xhats[t, :], parameter_shapes=self.parameter_shapes)
            params = vector_to_list_parameters(
                vec=x[t, :], parameter_shapes=self.parameter_shapes
            )
            # enable gradient
            for param in params:
                param.requires_grad_()
            # forward-pass using non-linear dynamics
            new_params, loss_temp, optimizer_sd_temp = self.train_one_minibatch_Adam(
                params=params,
                optimizer_sd=optimizer_sd_temp,
                minibatch=minibatches[t],
                eps_weight=u[t, :]
            )

            losses[t] = loss_temp

            with torch.no_grad():
                x_tp1 = torch.nn.utils.parameters_to_vector(parameters=new_params)

        return x, u, opt_sds, losses, x_tp1

    def iLQR(
        self,
        xhats: torch.Tensor,
        opt_sds: typing.List[dict],
        uhats: typing.Union[np.ndarray, torch.Tensor],
        minibatches: typing.List[typing.Dict[str, torch.Tensor]],
        num_iters: int,
        total_cost_hat: float
    ) -> typing.Tuple[torch.Tensor, torch.Tensor, typing.List[dict], np.ndarray, torch.Tensor]:
        """Iterative LQR to find approximate local optimal trajectory

        Args:
            xhats: nominal state
            opt_sds: list of Adam optimzer's state dictionaries corresponding to the nominal trajectory
            uhats: nominal action
            minibatches: data of many mini-batches
            num_iters: number of iterations used in iLQR
            loss_rand: the uniform-weighted validation loss induced by the nominal trajectory and negative prior of those uhats

        Returns:
            xhats: new nominal state
            uhats: new nominal action
            opt_sds: new list of Adam optimizer's state dictionaries corresponding to the new nominal trajectory
            losses: the validation losses of each mini-batch for monitoring purpose
        """
        opt_sds_temp = copy.deepcopy(opt_sds)
        for _ in range(num_iters):
            K, k = self.LQR_backward(
                xhats=xhats,
                opt_sds=opt_sds_temp,
                uhats=uhats,
                minibatches=minibatches
            )

            # perform forward-pass with line-search
            for alpha in [1, 0.5, 0.1]:

                x, u, opt_sds_temp, losses, x_tp1 = self.LQR_forward(
                    xhats=xhats,
                    optimizer_sd=opt_sds_temp[0],
                    uhats=uhats,
                    K=K,
                    k=k,
                    minibatches=minibatches,
                    alpha=alpha
                )

                if torch.isnan(input=u).any():
                    raise ValueError('u has NaN')

                # reset the attribute "step" in the Adam optimizer's state dictionary
                if (len(opt_sds_temp[0]['state']) > 0):
                    for i in range(len(opt_sds_temp)):
                        opt_sds_temp[i]['state'][0]['step'] = opt_sds_temp[0]['state'][0]['step'] + i

                break

                # calculate total cost
                total_cost = np.sum(losses) - torch.sum(input=self.normal_u.log_prob(value=u)).item()

                if (total_cost < total_cost_hat):
                    break

            # assign the newly-obtained trajectory as the nominal trajectory
            xhats = x
            uhats = u

        return xhats, uhats, opt_sds_temp[-1], losses, x_tp1
