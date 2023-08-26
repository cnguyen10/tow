import torch
# import random
import os
import csv
import typing
import numpy as np

from PIL import Image


def list_dir(root: str, prefix: bool = False) -> typing.List[str]:
    """List all directories at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = [p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))]
    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]
    return directories


def list_files(root: str, suffix: str, prefix: bool = False) -> typing.List[str]:
    """List all files ending with a suffix at a given root
    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = [p for p in os.listdir(root) if os.path.isfile(os.path.join(root, p)) and p.endswith(suffix)]
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


@torch.jit.script
def normalize_labels(labels: torch.Tensor) -> typing.Tuple[torch.Tensor, int]:
    """Normalize a list of labels, for example:
    [11, 11, 20, 20, 60, 60, 6, 6] => [0, 0, 1, 1, 2, 2, 3, 3]
    """
    if labels.ndim > 1:
        raise ValueError("Input must be a 1d tensor, not {}".format(labels.ndim))

    out = torch.empty_like(labels, dtype=labels.dtype, layout=labels.layout, device=labels.device, pin_memory=False, memory_format=torch.preserve_format)

    label_dict: typing.Dict[int, torch.Tensor] = {}
    for i in range(labels.numel()):
        val = labels[i].item()

        if val not in label_dict:
            label_dict[val] = torch.tensor(len(label_dict), device=labels.device)

        out[i] = label_dict[val]

    return out, len(label_dict)


@torch.jit.script
def train_val_split(eps_data: typing.List[torch.Tensor], k_shot: int, v_shot: int) -> typing.Dict[str, torch.Tensor]:
    """Split data into train and validation

    Args:
        eps_data: a list of 2 tensors:
            + first tensor: data
            + second tensor: labels
        k_shot: number of training data per class
        v_shot: number of validation data per class
        shuffle: shuffle data before splitting

    Returns: a dictionary containing data splitted
    """
    # get information of image size
    nc, iH, iW = eps_data[0][0].shape

    # get labels
    labels, num_classes = normalize_labels(labels=eps_data[1])

    # v_shot = int(labels.numel() / num_classes) - k_shot

    data = {
        'x_t': torch.empty(size=(num_classes, k_shot, nc, iH, iW), device=eps_data[0].device),
        'x_v': torch.empty(size=(num_classes, v_shot, nc, iH, iW), device=eps_data[0].device),
        'y_t': torch.empty(size=(num_classes * k_shot,), dtype=torch.int64, device=eps_data[1].device),
        'y_v': torch.empty(size=(num_classes * v_shot,), dtype=torch.int64, device=eps_data[1].device)
    }
    for cls_id in range(num_classes):
        X = eps_data[0][labels == cls_id][:(k_shot + v_shot)]
        data['x_t'][cls_id, :, :, :, :] = X[:k_shot]
        data['x_v'][cls_id, :, :, :, :] = X[k_shot:]

        data['y_t'][k_shot * cls_id: k_shot * (cls_id + 1)] = torch.as_tensor(data=[cls_id] * k_shot, dtype=torch.int64, device=labels.device)
        data['y_v'][v_shot * cls_id: v_shot * (cls_id + 1)] = torch.as_tensor(data=[cls_id] * v_shot, dtype=torch.int64, device=labels.device)

    data['x_t'] = data['x_t'].view(num_classes * k_shot, nc, iH, iW)
    data['x_v'] = data['x_v'].view(num_classes * v_shot, nc, iH, iW)

    return data


def get_episodes(episode_file_path: typing.Optional[str] = None, num_episodes: int = 100) -> typing.List[str]:
    """Get episodes from a file

    Args:
        episode_file_path:
        num_episodes: dummy variable in training to create an infinite
            episode (str) generator. In testing, it defines how many
            episodes to evaluate

    Return: an episode (str) generator
    """
    # get episode list if not None
    if episode_file_path is not None:
        episodes = []
        with open(file=episode_file_path, mode='r') as f_csv:
            csv_rd = csv.reader(f_csv, delimiter=',')
            episodes = list(csv_rd)
    else:
        episodes = [None] * num_episodes

    return episodes


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if m.weight is not None:
            torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias.data)
    elif classname.find('BatchNorm') != -1:
        if m.weight is not None:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)


def euclidean_distance(matrixN: torch.Tensor, matrixM: torch.Tensor) -> torch.Tensor:
    """Calculate Euclidean distance from N points to M points

    Args:
        matrixN: an N x D matrix for N points
        matrixM: a M x D matrix for M points

    Returns: N x M matrix
    """
    N = matrixN.size(0)
    M = matrixM.size(0)
    D = matrixN.size(1)
    assert D == matrixM.size(1)

    matrixN = matrixN.unsqueeze(1).expand(N, M, D)
    matrixM = matrixM.unsqueeze(0).expand(N, M, D)

    return torch.norm(input=matrixN - matrixM, p='fro', dim=2)


def get_cls_prototypes(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate the prototypes/centroids

    Args:
        x: input data
        y: corresponding labels

    Returns: a tensor of prototypes with shape (C, d),
        where C is the number of classes, d is the embedding dimension
    """
    _, d = x.shape
    cls_idx = torch.unique(input=y, return_counts=False)
    C = cls_idx.shape[0]

    prototypes = torch.empty(size=(C, d), device=x.device)
    for c in range(C):
        prototypes[c, :] = torch.mean(input=x[y == cls_idx[c]], dim=0)

    return prototypes


def kl_divergence_gaussians(p: typing.List[torch.Tensor], q: typing.List[torch.Tensor]) -> torch.Tensor:
    """Calculate KL divergence between 2 diagonal Gaussian

    Args: each paramter is list with 1st half as mean, and the 2nd half is log_std

    Returns: KL divergence
    """
    assert len(p) == len(q)

    n = len(p) // 2

    kl_div = 0
    for i in range(n):
        p_mean = p[i]
        p_log_std = p[n + i]

        q_mean = q[i]
        q_log_std = q[n + i]

        s1_vec = torch.exp(input=2 * q_log_std)
        mahalanobis = torch.sum(input=torch.square(input=p_mean - q_mean) / s1_vec)

        tr_s1inv_s0 = torch.sum(input=torch.exp(input=2 * (p_log_std - q_log_std)))

        log_det = 2 * torch.sum(input=q_log_std - p_log_std)

        kl_div_temp = mahalanobis + tr_s1inv_s0 + log_det - torch.numel(p_mean)
        kl_div_temp = kl_div_temp / 2

        kl_div = kl_div + kl_div_temp

    return kl_div


def vector_to_list_parameters(vec: torch.Tensor, parameter_shapes: typing.List[torch.Size]) -> torch.Tensor:
    """
    """
    params = []

    # Pointer for slicing the vector for each parameter
    pointer = 0

    for param_shape in parameter_shapes:
        # The length of the parameter
        num_param = np.prod(a=param_shape)

        params.append(vec[pointer:pointer + num_param].view(param_shape))

        # Increment the pointer
        pointer += num_param

    return params


def intialize_parameters(state_dict: dict) -> typing.List[torch.Tensor]:
    """"""
    p = list(state_dict.values())
    for m in p:
        if m.ndim > 1:
            torch.nn.init.kaiming_normal_(tensor=m, nonlinearity='relu')
        else:
            torch.nn.init.zeros_(tensor=m)

    return p


class IdentityNet(torch.nn.Module):
    """Identity hyper-net class for MAML"""
    def __init__(self, base_net: torch.nn.Module) -> None:
        super(IdentityNet, self).__init__()
        base_state_dict = base_net.state_dict()

        params = intialize_parameters(state_dict=base_state_dict)

        self.params = torch.nn.ParameterList([torch.nn.Parameter(p.float()) for p in params])
        self.identity = torch.nn.Identity()

    def forward(self) -> typing.List[torch.Tensor]:
        out = []
        for param in self.params:
            temp = self.identity(param)
            out.append(temp)
        return out


class NormalVariationalNet(torch.nn.Module):
    """A simple neural network that simulate the
    reparameterization trick. Its parameters are
    the mean and std-vector
    """
    def __init__(self, base_net: torch.nn.Module) -> None:
        """
        Args:
            base_net: the base network
        """
        super(NormalVariationalNet, self).__init__()

        # dict of parameters of based network
        base_state_dict = base_net.state_dict()

        mean = intialize_parameters(state_dict=base_state_dict)

        # initialize parameters
        self.mean = torch.nn.ParameterList([torch.nn.Parameter(m) for m in mean])
        self.log_std = torch.nn.ParameterList([torch.nn.Parameter(torch.rand_like(v) - 4) for v in base_state_dict.values()])

        self.num_base_params = np.sum([torch.numel(p) for p in self.mean])

    def forward(self) -> typing.List[torch.Tensor]:
        """Output the parameters of the base network in list format to pass into higher monkeypatch
        """
        out = []
        for m, log_s in zip(self.mean, self.log_std):
            eps_normal = torch.randn_like(m, device=m.device)
            temp = m + eps_normal * torch.exp(input=log_s)
            out.append(temp)
        return out


def pil_loader_color(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def pil_loader_gray(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        img.load()
        return img
