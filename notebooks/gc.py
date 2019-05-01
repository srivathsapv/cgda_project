import os

import time
# import progressbar
import warnings
import copy
import pandas as pd
import numpy as np

import torch
import torch.utils.data as utils
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from multiprocessing import cpu_count
from torch.multiprocessing import Pool

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

BATCH_SIZE = 32
EPOCH = 25
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_data_from_csv(path):
    df = pd.read_csv(path)
    X = df["sequence"].values
    y = df["label"].values
    return X, y


def load_train_val_test_data(level, analyze=True):
    data_base_path = "./data/hierarchy/" + level
    train_sequences, train_labels = read_data_from_csv(os.path.join(data_base_path, "train.csv"))
    val_sequences, val_labels = read_data_from_csv(os.path.join(data_base_path, "val.csv"))
    test_sequences, test_labels = read_data_from_csv(os.path.join(data_base_path, "test.csv"))

    if analyze:
        a = list(map(lambda x: len(x), train_sequences))
        print("DNA Sequence Length Statistics:")
        print("Max:", np.max(a))
        print("Min:", np.min(a))
        print("Mean:", np.ceil(np.mean(a)))
        print("Median:", np.ceil(np.median(a)))
        print("Sqrt of Max:", np.ceil(np.sqrt(np.max(a))))

    return train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels


IMAGE_WIDTH = IMAGE_HEIGHT = 21  # 441 length zero-padded DNA sequences
IMAGE_CHANNELS = 4  # A, C, G, T

base_pair_map = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'X': [0, 0, 0, 0]
}


def seqeunces_to_image(sequences):
    image = np.zeros((len(sequences), IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS))
    for i, sequence in enumerate(sequences):
        for loc, base_pair in enumerate(sequence):
            row = loc // IMAGE_HEIGHT
            col = loc % IMAGE_HEIGHT
            image[i, row, col] = base_pair_map[base_pair]
    return image


# analyze sequences to get the image size
train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels = load_train_val_test_data(
    "phylum")

base_pair_colors = {
    (1, 0, 0, 0): [183, 28, 28],  # red
    (0, 1, 0, 0): [174, 234, 0],  # green
    (0, 0, 1, 0): [0, 145, 234],  # blue
    (0, 0, 0, 1): [255, 111, 0],  # orange
    (0, 0, 0, 0): [33, 33, 33]   # black
}

base_pair_char = {
    (1, 0, 0, 0): "A",
    (0, 1, 0, 0): "C",
    (0, 0, 1, 0): "G",
    (0, 0, 0, 1): "T",
    (0, 0, 0, 0): "X"
}


def create_pytorch_datasets(data, labels):
    tensor_x = torch.stack([torch.Tensor(np.swapaxes(i, 0, 2))
                            for i in data])  # transform to torch tensors
    tensor_y = torch.stack([torch.Tensor([i]) for i in labels]).long().view(-1)

    dataset = utils.TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = utils.DataLoader(dataset, batch_size=BATCH_SIZE)  # create your dataloader

    return dataloader


class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(5*5*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = F.log_softmax(self.fc(out), dim=1)
        return out


def cnn_train_model(model, train_loader, test_loader, optimizer, EPOCH):
    model = model.to(DEVICE)

    if DEVICE == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    t0 = time.perf_counter()

    loss_train = np.zeros((EPOCH,))
    loss_test = np.zeros((EPOCH,))
    acc_test = np.zeros((EPOCH,))
    acc_train = np.zeros((EPOCH,))
    time_test = np.zeros((EPOCH,))

    # bar = progressbar.ProgressBar(min_value=1, max_value=EPOCH)
    for epoch in range(EPOCH):
        # bar.update(epoch+1)

        # train 1 epoch
        model.train()
        correct = 0
        train_loss = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            b_x = Variable(x)
            b_y = Variable(y)
            scores = model(b_x)
            loss = F.nll_loss(scores, b_y)      # negative log likelyhood
            optimizer.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer.step()                    # apply gradients
            model.zero_grad()

            # computing training accuracy
            pred = scores.data.max(1, keepdim=True)[1]
            correct += pred.eq(b_y.data.view_as(pred)).long().cpu().sum()
            train_loss += F.nll_loss(scores, b_y, reduction='sum').item()

        acc_train[epoch] = 100 * float(correct) / float(len(train_loader.dataset))
        loss_train[epoch] = train_loss / len(train_loader.dataset)

        # testing
        model.eval()
        correct = 0
        test_loss = 0
        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            b_x = Variable(x)
            b_y = Variable(y)
            scores = model(b_x)
            test_loss += F.nll_loss(scores, b_y, reduction='sum').item()
            pred = scores.data.max(1, keepdim=True)[1]
            correct += pred.eq(b_y.data.view_as(pred)).long().cpu().sum()

        loss_test[epoch] = test_loss/len(test_loader.dataset)
        acc_test[epoch] = 100 * float(correct) / float(len(test_loader.dataset))
        time_test[epoch] = time.perf_counter() - t0

    return [acc_train, acc_test, loss_train, loss_test]


def cnn_train_eval(level, model, eval_on="test", cnn_config={"lr": 0.001, "weight_decay": 0}):
    # load train-test data and convert to a PyTorch Dataset of QRCode images
    train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels = load_train_val_test_data(
        level, analyze=False)
    train_loader = create_pytorch_datasets(seqeunces_to_image(train_sequences), train_labels)
    if eval_on == "test":
        eval_loader = create_pytorch_datasets(seqeunces_to_image(test_sequences), test_labels)
    elif eval_on == "val":
        eval_loader = create_pytorch_datasets(seqeunces_to_image(val_sequences), val_labels)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cnn_config["lr"], weight_decay=cnn_config["weight_decay"])
    logs = cnn_train_model(model, train_loader, eval_loader, optimizer, EPOCH)

    output = logs + [cnn_config]
    print(output)

    return output


levels_and_models = [("phylum", ConvNet(3)), ("class", ConvNet(5)), ("order", ConvNet(10))]
lr_space = np.geomspace(1e-6, 1e3, num=10)
weight_decay = np.geomspace(1e-6, 1e3, num=10)


# populate paramter dicts
param_dicts = list()
for model_id, (level, m) in enumerate(levels_and_models):
    for l in lr_space:
        for w in weight_decay:
            param_dict = {"level": level, "model": copy.deepcopy(m), "eval_on": "val", "cnn_config": {
                "model": model_id, "lr": l, "weight_decay": w}}
            param_dicts.append(param_dict)


def cnn_train_test_unpack(args):
    return cnn_train_eval(**args)


with Pool(int(cpu_count() / torch.get_num_threads()) - 1) as p:
    experiment_logs = p.map(cnn_train_test_unpack, param_dicts)
np.save("grid_search_best_cnn_logs.npy", np.array(experiment_logs))
