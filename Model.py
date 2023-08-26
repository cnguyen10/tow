import torch
import typing
from copy import deepcopy


class Model(object):
    """Class that stores parameters of a model
    """
    def __init__(self, params: typing.List[torch.Tensor], config: dict, requires_grad: bool = False, **kwargs) -> None:
        """Initialize an instance of a model
        Args:
            params: a list consisting the paramters of the model
            state_dict: a dictionary consisting the state of the model's optimizer
        """
        if (kwargs['name'] != 'tow'):
            device = config['device2']
        else:
            device = config['device']

        with torch.no_grad():
            self.params = [param.clone().to(device) for param in params]

        if ('state_dict' in kwargs):
            self.state_dict = deepcopy(kwargs['state_dict'])
            self.state_dict['param_groups'][0]['lr'] = config['meta_lr']
            self.state_dict['param_groups'][0]['weight_decay'] = config['weight_decay']
        else:
            self.state_dict = torch.optim.Adam(params=self.params, lr=config['meta_lr'], weight_decay=config['weight_decay']).state_dict()

        self.requires_grad = requires_grad

        if requires_grad:
            for param in self.params:
                param.requires_grad_()

        return None
