import warnings

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from ml.model.cnn.activation_map_viz import load_pretrained_model, viz_activations_for_all
from ml.model.cnn.data_loader import (create_pytorch_datasets,
                                      load_data_from_dump)
from ml.utils import get_logger

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

LOGGER = get_logger()


def cnn_test(model, test_loader, config):
    DEVICE = config["device"]

    model = model.to(DEVICE)

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

    loss_test = test_loss / len(test_loader.dataset)
    acc_test = 100 * float(correct) / \
        float(len(test_loader.dataset))

    return [acc_test, loss_test]


def cnn_test_model(level, path_config, cnn_config={"batch_size": 32, "device": "cpu"}):
    (_, _), (_, _), (test_images, test_labels) = load_data_from_dump(
        level, path_config["input_path"])

    model = load_pretrained_model(level, path_config)
    eval_loader = create_pytorch_datasets(test_images, test_labels, cnn_config)

    logs = cnn_test(model, eval_loader, cnn_config)

    LOGGER.info(level.capitalize() + "-level Classifier")
    LOGGER.info("Test Accuracy:" + str(logs[0]))
    LOGGER.info("Test Loss:" + str(logs[1]) + "\n")


def cnn_test_all_models(path_config):
    LOGGER.info("Testing the pre-trained models checkpointed after running train_model.py (uses the test data split to evaluate) ...\n")
    for level in ["phylum", "class", "order"]:
        cnn_test_model(level, path_config)

    viz_activations_for_all(path_config)


if __name__ == '__main__':
    path_config = {"hierarchy_path": "data/hierarchy/", "input_path": "./data/cnn/", "plots_path": "./results/cnn/plots/",
                   "models_path": "./results/cnn/models/", "grid_search_path": "./results/cnn/grid_search/", "activations_path": "results/cnn/activations/"}
    cnn_test_all_models(path_config)
