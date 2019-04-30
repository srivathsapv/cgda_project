import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data as utils

import ml.utils as utils

plt.style.use('seaborn')

warnings.filterwarnings("ignore")
torch.set_num_threads(1)


def read_data_from_csv(path):
    df = pd.read_csv(path)
    X = df["sequence"].values
    y = df["label"].values
    y_names = df["class_name"].values
    return X, y, y_names


def load_train_val_test_data(base_path, level, analyze=True, return_label_names=False):
    base_path_on_level = os.path.join(base_path, level)
    train_sequences, train_labels, train_label_names = read_data_from_csv(
        os.path.join(base_path_on_level, "train.csv"))
    val_sequences, val_labels, val_label_names = read_data_from_csv(
        os.path.join(base_path_on_level, "val.csv"))
    test_sequences, test_labels, test_label_names = read_data_from_csv(
        os.path.join(base_path_on_level, "test.csv"))

    if analyze:
        a = list(map(lambda x: len(x), train_sequences))
        print("DNA Sequence Length Statistics:")
        print("Max:", np.max(a))
        print("Min:", np.min(a))
        print("Mean:", np.ceil(np.mean(a)))
        print("Median:", np.ceil(np.median(a)))
        print("Sqrt of Max:", np.ceil(np.sqrt(np.max(a))))
    if return_label_names:
        return train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels, test_label_names
    return train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels


def create_pytorch_datasets(data, labels, config):
    tensor_x = torch.stack([torch.Tensor(np.swapaxes(i, 0, 2))
                            for i in data])  # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor([i]) for i in labels]).long().view(-1)

    dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = utils.DataLoader(dataset, batch_size=config["batch_size"])  # create your dataloader

    return dataloader
