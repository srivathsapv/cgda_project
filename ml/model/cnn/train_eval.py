import os
import time
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
from torchviz import make_dot
from tqdm import tqdm

from ml.model.cnn.architecture import ConvNet
from ml.model.cnn.data_loader import (create_pytorch_datasets,
                                      load_data_from_dump)
from ml.model.cnn.plot import plot_grid_search, plot_train_eval_curves
from ml.utils import get_logger

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

LOGGER = get_logger()


def cnn_train_model(model, train_loader, test_loader, optimizer, config):
    EPOCH = config["epoch"]
    DEVICE = config["device"]

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

    for epoch in tqdm(range(EPOCH), total=EPOCH):

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

        acc_train[epoch] = 100 * \
            float(correct) / float(len(train_loader.dataset))
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

        loss_test[epoch] = test_loss / len(test_loader.dataset)
        acc_test[epoch] = 100 * float(correct) / \
            float(len(test_loader.dataset))
        time_test[epoch] = time.perf_counter() - t0

    return [acc_train, acc_test, loss_train, loss_test, model]


def cnn_train_eval(level, model, path_config, eval_on="test", cnn_config={
                   "lr": 0.001, "weight_decay": 0, "epoch": 25, "batch_size": 32, "device": "cpu"}, is_plot=True, save_model=False):
    (train_images, train_labels), (val_images, val_labels), (test_images,
                                                             test_labels) = load_data_from_dump(level, path_config["input_path"])

    train_loader = create_pytorch_datasets(
        train_images, train_labels, cnn_config)

    if eval_on == "test":
        eval_loader = create_pytorch_datasets(
            test_images, test_labels, cnn_config)
    elif eval_on == "val":
        eval_loader = create_pytorch_datasets(
            val_images, val_labels, cnn_config)

    optimizer = torch.optim.Adam(model.parameters(
    ), lr=cnn_config["lr"], weight_decay=cnn_config["weight_decay"])
    logs = cnn_train_model(model, train_loader,
                           eval_loader, optimizer, cnn_config)

    LOGGER.info("\nTrain Accuracy:" + str(logs[0][-1]))
    LOGGER.info("Train Loss:" + str(logs[2][-1]))
    LOGGER.info("Test Accuracy:" + str(logs[1][-1]))
    LOGGER.info("Test Loss:" + str(logs[3][-1]) + "\n")

    if save_model:
        if not os.path.exists(path_config["models_path"]):
            os.makedirs(path_config["models_path"])
        torch.save(logs[-1].state_dict(), os.path.join(
            path_config["models_path"], "best_" + level + "_cnn_model"))

    if is_plot:
        if not os.path.exists(os.path.join(path_config["plots_path"], level)):
            os.makedirs(os.path.join(path_config["plots_path"], level))
        plot_train_eval_curves(
            *(logs[:-1] + [os.path.join(path_config["plots_path"], level, "train_" + eval_on + "_loss_acc.jpg")]))

    output = logs[:-1] + [{**cnn_config, **{"trained_model": logs[-1]}}]

    return output


def plot_archs(path_config):
    # Print the CNN architecture
    models = [ConvNet(3), ConvNet(5), ConvNet(10)]

    for model, level in zip(models, ["phylum", "class", "order"]):
        x = torch.zeros(1, 4, 21, 21, dtype=torch.float, requires_grad=True)
        out = model(x)
        dot = make_dot(out, params=dict(
            list(model.named_parameters()) + [('Input Image', x)]))

        dot.format = 'png'
        dot.render(os.path.join(
            path_config["models_path"], level+"_model_cnn_arch"))


def train_best_cnn_models(cnn_config, path_config,
                          is_demo=False, save_model=True, is_plot=True):
    LOGGER.info(
        "Visualizations of the CNN architectures have been saved to: path_config['models_path']\n")
    plot_archs(path_config)

    LOGGER.info(
        "Training with the best possible params and evaluating on validation and test datasets ...\n")

    LOGGER.info("NOTE: The best possible parameters were found using the grid_search.py script file." +
                " This tests 300 different settings and ran on a 40 core machine for 2 hours." +
                " To see the grid search 3D plots over weight_decay and learning_rate," +
                " go to path_config['grid_search_results'].\n")

    LOGGER.info(
        "NOTE: Run grid_search.py if you want to run the grid search from scratch.\n")
    plot_grid_search(path_config)

    if is_demo:
        LOGGER.info(
            "WARNING: Runnning in DEMO mode. (Only 2 epochs will be run and the trained model will not be saved)\n")
        for level in ["phylum", "class", "order"]:
            cnn_config[level]["epoch"] = 2
        save_model = False
        is_plot = False

    LOGGER.info("Phylum-level CNN Classifier Training ...")
    cnn_train_eval("phylum", ConvNet(3), path_config, eval_on="val",
                   cnn_config=cnn_config["phylum"], is_plot=is_plot, save_model=False)
    cnn_train_eval("phylum", ConvNet(3), path_config, eval_on="test",
                   cnn_config=cnn_config["phylum"], is_plot=is_plot, save_model=save_model)

    LOGGER.info("Class-level CNN Classifier Training ...")
    cnn_train_eval("class", ConvNet(5), path_config, eval_on="val",
                   cnn_config=cnn_config["class"], is_plot=is_plot, save_model=False)
    cnn_train_eval("class", ConvNet(5), path_config, eval_on="test",
                   cnn_config=cnn_config["class"], is_plot=is_plot, save_model=save_model)

    LOGGER.info("Order-level CNN Classifier Training ...")
    cnn_train_eval("order", ConvNet(10), path_config, eval_on="val",
                   cnn_config=cnn_config["order"], is_plot=is_plot, save_model=False)
    cnn_train_eval("order", ConvNet(10), path_config, eval_on="test",
                   cnn_config=cnn_config["order"], is_plot=is_plot, save_model=save_model)


if __name__ == '__main__':
    path_config = {"input_path": "./data/cnn/", "plots_path": "./results/cnn/plots/",
                   "models_path": "./results/cnn/models/", "grid_search_path": "./results/cnn/grid_search/"}

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cnn_config = {
        "phylum": {"lr": 0.01, "weight_decay": 1e-05, "epoch": 25, "batch_size": 32, "device": DEVICE},
        "class": {"lr": 0.01, "weight_decay": 0.0001, "epoch": 25, "batch_size": 32, "device": DEVICE},
        "order": {"lr": 0.01, "weight_decay": 0.0001, "epoch": 25, "batch_size": 32, "device": DEVICE}
    }

    train_best_cnn_models(cnn_config, path_config, is_demo=True)
