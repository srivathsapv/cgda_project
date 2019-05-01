import copy
import os
import warnings
from multiprocessing import cpu_count

import numpy as np
import torch
from torch.multiprocessing import Pool

from ml.model.cnn.architecture import ConvNet
from ml.model.cnn.train_eval import cnn_train_eval

warnings.filterwarnings("ignore")
torch.set_num_threads(1)


def generate_search_space_configurations(path_config):
    # search space
    levels_and_models = [("phylum", ConvNet(3)), ("class", ConvNet(5)), ("order", ConvNet(10))]
    lr_space = np.geomspace(1e-6, 1e3, num=10)
    weight_decay = np.geomspace(1e-6, 1e3, num=10)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # populate paramter dicts
    param_dicts = list()
    for model_id, (level, m) in enumerate(levels_and_models):
        for l in lr_space:
            for w in weight_decay:
                param_dict = {"level": level, "model": copy.deepcopy(m), "path_config": path_config, "eval_on": "val", "cnn_config": {
                    "model": model_id, "lr": l, "weight_decay": w, "epoch": 25, "batch_size": 32, "device": DEVICE}, "is_plot": False, "save_model": False}
                param_dicts.append(param_dict)

    return param_dicts


# search for best params in parallel
def cnn_train_test_unpack(args):
    return cnn_train_eval(**args)


def run_grid_search_parallel(path_config):
    param_dicts = generate_search_space_configurations(path_config)

    with Pool(int(cpu_count() / torch.get_num_threads()) - 1) as p:
        experiment_logs = p.map(cnn_train_test_unpack, param_dicts[:6])

    if not os.path.exists(path_config["grid_search_path"]):
        os.makedirs(path_config["grid_search_path"])

    np.save(os.path.join(path_config["grid_search_path"], "grid_search_best_cnn_logs.npy"), np.array(experiment_logs))


if __name__ == '__main__':
    path_config = {"input_path": "./data/cnn_qrcode/", "plots_path": "./results/cnn_qrcode/plots/", "models_path": "./results/cnn_qrcode/models/", "grid_search_path": "./results/cnn_qrcode/grid_search/"}
    run_grid_search_parallel(path_config)
