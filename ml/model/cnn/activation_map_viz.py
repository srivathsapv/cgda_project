import os
import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm

from ml.model.cnn.architecture import ConvNet
from ml.model.cnn.data_loader import (load_data_from_dump,
                                      load_train_val_test_data)
from ml.model.cnn.guided_backprop import GuidedBackprop
from ml.utils import get_logger

plt.style.use('seaborn')

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

LOGGER = get_logger()


def convert_gradient_to_image(gradient):
    # Normalize
    gradient = gradient - gradient.min()
    gradient /= gradient.max()

    if gradient.shape[0] == 3 and np.max(gradient) == 1:
        gradient = gradient.transpose(1, 2, 0) * 255
    elif gradient.shape[0] == 3 and np.max(gradient) > 1:
        gradient = gradient.transpose(1, 2, 0)

    gradient = gradient.astype(np.uint8)
    gradient = Image.fromarray(gradient)

    with BytesIO() as f:
        gradient.save(f, format='JPEG')
        f.seek(0)
        return np.array(Image.open(BytesIO(f.read())))


def get_positive_negative_saliency(gradient):
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency


def load_data_for_activations_viz(level, path_config, num_samples_per_class=10):
    test_data, test_labels = load_data_from_dump(
        level, path_config["input_path"])[-1]
    test_label_names = load_train_val_test_data(
        path_config["hierarchy_path"], level, analyze=False, return_label_names=True)[-1]

    data = dict()
    label_map = dict()
    for label in np.unique(test_labels):
        choosen = np.random.choice(np.argwhere(test_labels == label).flatten(),
                                   num_samples_per_class, replace=False)
        sub_data = test_data[choosen]
        label_name = np.unique(test_label_names[choosen])[0]
        data[label] = sub_data
        label_map[label] = label_name
    return data, label_map


def load_pretrained_model(level, path_config):
    if level == "phylum":
        model = ConvNet(3)
        model.load_state_dict(torch.load(os.path.join(
            path_config["models_path"], "best_phylum_cnn_model")))
    elif level == "class":
        model = ConvNet(5)
        model.load_state_dict(torch.load(os.path.join(
            path_config["models_path"], "best_class_cnn_model")))
    elif level == "order":
        model = ConvNet(10)
        model.load_state_dict(torch.load(os.path.join(
            path_config["models_path"], "best_order_cnn_model")))

    return model


def get_activation_map(input_image, level, class_id, path_config):
    # prepare input image
    input_image = input_image.transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).float()
    input_image.unsqueeze_(0)
    input_image = Variable(input_image, requires_grad=True)

    # Init Guided Backprop
    model = load_pretrained_model(level, path_config)
    GBP = GuidedBackprop(model)

    # Get Gradients (clip to just 3 of 4 channels)
    guided_gradients = GBP.generate_gradients(input_image, class_id)[:3, :, :]

    # Negative Saliency Maps
    _, neg_sal = get_positive_negative_saliency(guided_gradients)
    activation_map_image = convert_gradient_to_image(_)

    return activation_map_image


def plot_activation_maps(level, class_id, class_name, data, path_config):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.grid(b=None)

    rc = int(np.ceil(np.sqrt(len(data))))
    for i, d in enumerate(data):
        image = get_activation_map(d, level, class_id, path_config)
        ax = fig.add_subplot(rc, rc, i+1)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.imshow(image)

    fig.suptitle(level.capitalize() + "-level Activations (1st layer) for " +
                 class_name, fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = os.path.join(path_config["activations_path"], level)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, class_name + ".png"))
    plt.clf()


def viz_activations_for(level, path_config, num_samples_per_class=25):
    category_wise_data, categories = load_data_for_activations_viz(
        level, path_config, num_samples_per_class)
    for c in tqdm(categories, total=len(categories)):
        plot_activation_maps(
            level, c, categories[c], category_wise_data[c], path_config)


def viz_activations_for_all(path_config):
    LOGGER.info(
        "Visualizing the First Convolution Layer of the CNN using GuidedBackprop ...")
    LOGGER.info(
        "Loading the pre-trained model from: path_config['models_path'] and saving the activation maps to: path_config['activations_path']\n")
    for level in ["phylum", "class", "order"]:
        LOGGER.info("Generating and saving visualizations at a " +
                    level + "-level ...")
        viz_activations_for(level, path_config)


if __name__ == '__main__':
    path_config = {"input_path": "./data/cnn/", "plots_path": "./results/cnn/plots/",
                   "models_path": "./results/cnn/models/", "grid_search_path": "./results/cnn/grid_search/",
                   "activations_path": "./results/cnn/activations/", "hierarchy_path": "./data/hierarchy"}

    viz_activations_for_all(path_config)
