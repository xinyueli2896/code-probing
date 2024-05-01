import datetime
import os
import torch
import argparse

def config(task):
    layers = list((range(13)))
    n_layers = 13
    size_dataset = 7000 if task == 'relational' else 10000
    hidden_size = 768
    return layers, n_layers, size_dataset, hidden_size
    
def timestamp():
    return datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")

def load_activation_probing_dataset(layer_ix, task):
  save_name = f'{layer_ix}.pt'
  save_path = os.path.join(f"./{task}/activations", save_name)
  activations = torch.load(save_path)
  return activations

def parse_arguments():
    """
    Parses command-line arguments using argparse and returns the parsed arguments.
    
    Returns:
    Namespace: An argparse.Namespace object containing the arguments and their values.
    """
    parser = argparse.ArgumentParser()
    # Required positional argument
    parser.add_argument('--task', type=str,
                        help='can either be relational or operational')
    parser.add_argument('--tag', type=str, default = datetime.datetime.now().strftime("%Y:%m:%d"),
                        help='identifier for skpts')
    parser.add_argument('--probe_type', type=str, default='Linear',
                        help='Linear/NonLinear')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='num of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    args = parser.parse_args()
    return args