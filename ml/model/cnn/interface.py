import torch
from ml.model.cnn.train_eval import train_best_cnn_models
import ml.utils as utils

RUN_OPTIONS = ["cnn_qrcode"]


def train_model(path_config, args):

    path_config = path_config["cnn_qrcode"]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_config = utils.get_model_hyperparams("cnn_qrcode")
    for level in ["phylum", "class", "order"]:
        cnn_config[level]["device"] = DEVICE

    train_best_cnn_models(cnn_config, path_config, is_demo=args.is_demo)


def test_model(path_config, args):
    path_config = path_config["cnn_qrcode"]

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cnn_config = utils.get_model_hyperparams("cnn_qrcode")
    for level in ["phylum", "class", "order"]:
        cnn_config[level]["device"] = DEVICE
