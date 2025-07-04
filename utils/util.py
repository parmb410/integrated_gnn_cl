# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import random
import numpy as np
import torch
import sys
import os
import argparse
import torchvision
import PIL


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append('eval%d_in' % i)
            eval_name_dict['valid'].append('eval%d_out' % i)
        else:
            eval_name_dict['target'].append('eval%d_out' % i)
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'diversify': ['class', 'dis', 'total']}
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def act_param_init(args):
    args.select_position = {'emg': [0], 'dsads': [0]}
    args.select_channel = {'emg': np.arange(8), 'dsads': np.arange(45)}
    args.hz_list = {'emg': 1000, 'dsads': 25}
    args.act_people = {
        'emg': [[i*9+j for j in range(9)]for i in range(4)],
        'dsads': [[i*8+j for j in range(8)]for i in range(8)]  # Example
    }
    tmp = {
        'emg': ((8, 1, 200), 6, 10),
        'dsads': ((45, 1, 125), 19, 5)  # <-- example config, update as needed
    }
    if args.dataset not in tmp:
        raise ValueError(
            f"Dataset '{args.dataset}' not supported in act_param_init(). "
            f"Supported datasets are: {list(tmp.keys())}.\n"
            f"Either use --dataset emg or add your dataset config to act_param_init()."
        )
    args.input_shape, args.num_classes, args.grid_size = tmp[args.dataset]
    return args

# --------- ADDED: Robust extraction of domain labels from dataset ----------
def extract_domain_labels(dataset):
    """Try to extract a list of domain labels from a dataset."""
    if hasattr(dataset, 'domain_labels'):
        return dataset.domain_labels
    try:
        # If dataset is indexable, try to get third item from each __getitem__
        domain_labels = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            if isinstance(item, (tuple, list)) and len(item) >= 3:
                domain_labels.append(item[2])
            else:
                raise Exception
        return domain_labels
    except Exception:
        raise ValueError("Cannot extract domain_labels from dataset. Please ensure your dataset "
                         "has a .domain_labels attribute or returns (x, y, domain_label) for each sample.")

# -------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description='DG')
    parser.add_argument('--algorithm', type=str, default="diversify")
    parser.add_argument('--alpha', type=float,
                        default=0.1, help="DANN dis alpha")
    parser.add_argument('--alpha1', type=float,
                        default=0.1, help="DANN dis alpha")
    parser.add_argument('--batch_size', type=int,
                        default=32, help="batch_size")
    parser.add_argument('--beta1', type=float, default=0.5, help="Adam")
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--checkpoint_freq', type=int,
                        default=100, help='Checkpoint every N steps')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='')
    parser.add_argument('--dataset', type=str, default='dsads')
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--dis_hidden', type=int, default=256)
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")
    parser.add_argument('--layer', type=str, default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--lam', type=float, default=0.0)
    parser.add_argument('--latent_domain_num', type=int, default=3)
    parser.add_argument('--local_epoch', type=int,
                        default=1, help='local iterations')
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--max_epoch', type=int,
                        default=120, help="max iterations")
    parser.add_argument('--model_size', default='median',
                        choices=['small', 'median', 'large', 'transformer'])
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--old', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=str, default="cross_people")
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output', type=str, default="train_output")
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--use_gnn', type=int, default=0, help="Enable GNN feature extractor if 1")
    parser.add_argument('--curriculum', action='store_true', help='Enable curriculum learning')
    parser.add_argument('--curriculum_step', type=int, default=5, help='Epochs per curriculum stage')
    args = parser.parse_args()
    args.steps_per_epoch = 10000000000
    args.data_dir = args.data_file+args.data_dir
    os.environ['CUDA_VISIBLE_DEVICS'] = args.gpu_id
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = act_param_init(args)
    return args
