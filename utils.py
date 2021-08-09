import json
import os
import torch
import torch.backends.cudnn as cudnn
import random
import random as np
from datetime import datetime


def read_json(path):
    """
    Read JSON file from path
    :param (string) path: path to he JSON file
    :return: data from the JSON file
    """
    with open(path) as f:
        data = json.load(f)
    f.close()

    return data


def save_json(path, data_obj):
    """
    Save data in the JSON format
    :param (string) path: path where to save JSON file
    :param (Any) data_obj: object with data
    :return:
    """
    with open(path, "w") as f:
        json.dump(data_obj, f, indent=4)
    f.close()


def human_format(num):
    """
    Convert number to human readable format
    :param (float) num: number
    :return (string): string number
    """
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0

    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])


def create_run_folder(path):
    """
    Create folder for saving Tnesorboard events in the deep_homography net
    :param (string) path: Path to the folder
    :return (string): string of the name of folder
    """
    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    folder = "run_" + dt_string
    run_folder_path = os.path.join(path, folder)
    os.mkdir(run_folder_path)   # create run folder
    os.mkdir(os.path.join(run_folder_path, "weights"))  # create weights folder
    print("[INFO] Run folder created")

    return folder


def calculate_acc(outputs, labels):
    """
    Calculate accuracy for one batch
    :param outputs: model predictions
    :param labels: ground truth labels
    :return: accuracy value from 0 to 1
    """
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == labels).sum().item()

    return correct / labels.size(0)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def init_torch_seeds(seed=0):
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.benchmark, cudnn.deterministic = False, True
    else:  # faster, less reproducible
        cudnn.benchmark, cudnn.deterministic = True, False
