import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from ml.utils import get_logger

plt.style.use('seaborn')

warnings.filterwarnings("ignore")

LOGGER = get_logger()


def plot_train_eval_curves(
        acc_train, acc_test, loss_train, loss_test, save_path):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(acc_train, label='Acc. Train')
    plt.plot(acc_test, label='Acc. Test')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss_train, label='Loss Train')
    plt.plot(loss_test, label='Loss Test')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.savefig(save_path)
    plt.clf()


def get_items_to_plot(model):
    acc_train, acc_test, loss_train, loss_test, configs = list(zip(*model))

    def get_final(x): return list(zip(*x))[-1]

    train_accuracies = get_final(acc_train)
    test_accuracies = get_final(acc_test)
    train_losses = get_final(loss_train)
    test_losses = get_final(loss_test)

    learning_rates, weight_decays = list(
        zip(*[(config["lr"], config["weight_decay"]) for config in configs]))

    return train_accuracies, test_accuracies, train_losses, test_losses, learning_rates, weight_decays


def remove_outliers(data, others, m=2, n=10):
    # criteria = abs(data - np.mean(data)) < m * np.std(data)
    criteria = data < n

    data[np.invert(criteria)] = max(data[criteria])

    truncated_others = list()
    for o in others:
        # o[np.invert(criteria)] = max(o[criteria])
        truncated_others.append(o)
    return data, truncated_others


def plot_performance_metric(metric, learning_rates, weight_decays,
                            z_label, title, path_config, clip_outliers=False):
    param_to_id = {n: float(i) for i, n in enumerate(
        np.geomspace(1e-6, 1e3, num=10))}

    fig = plt.figure(figsize=(14, 10))
    ax = fig.gca(projection='3d')

    X, Y = [param_to_id[l] for l in learning_rates], [param_to_id[w]
                                                      for w in weight_decays]

    if clip_outliers:
        metric, (X, Y) = remove_outliers(
            np.array(metric), others=[np.array(X), np.array(Y)])

    # # changes for plot_surface()
    # params_to_metric = {(param_to_id[l], param_to_id[w]): m for m, l, w in zip(metric, learning_rates, weight_decays)}
    # X, Y = np.meshgrid(X, Y)
    # zs = np.array([params_to_metric[(x, y)] for x, y in zip(np.ravel(X), np.ravel(Y))])
    # Z = zs.reshape(X.shape)

    ax.plot_trisurf(X, Y, metric, cmap='viridis', edgecolor='none')

    ax.set_title(title + " (" + z_label + ")",
                 fontsize=15, pad=15, weight='bold')
    ax.tick_params(labelsize=13)

    ax.xaxis.set_ticks(np.arange(10))
    ax.xaxis.set_ticklabels(np.geomspace(1e-6, 1e3, num=10))
    ax.set_xlabel('Learning Rate', labelpad=15, fontsize=13)

    ax.yaxis.set_ticks(np.arange(10))
    ax.yaxis.set_ticklabels(np.geomspace(1e-6, 1e3, num=10))
    ax.set_ylabel('Weight Decay', labelpad=15, fontsize=13)

    ax.set_zlabel(z_label, labelpad=15, fontsize=13)

    level = title.split("-")[0].lower()
    if not os.path.exists(os.path.join(
            path_config["grid_search_path"], "search_3d_plots", level)):
        os.makedirs(os.path.join(
            path_config["grid_search_path"], "search_3d_plots", level))

    plt.savefig(os.path.join(path_config["grid_search_path"], "search_3d_plots", level,
                             level + "_classifier_" + "_".join([str(i).lower() for i in z_label.split()]) + ".jpg"))


def segregate_models(experiment_logs):
    phylum_model, class_model, order_model = [
        [i for i in experiment_logs if i[4]["model"] == x] for x in [0, 1, 2]]

    # sort based on best test accuracy
    phylum_model = sorted(phylum_model, key=lambda x: x[1][-1], reverse=True)
    class_model = sorted(class_model, key=lambda x: x[1][-1], reverse=True)
    order_model = sorted(order_model, key=lambda x: x[1][-1], reverse=True)

    return phylum_model, class_model, order_model


def plot_grid_search_plots(model, title, path_config):
    train_accuracies, test_accuracies, train_losses, test_losses, learning_rates, weight_decays = get_items_to_plot(
        model)
    plot_performance_metric(train_accuracies, learning_rates, weight_decays,
                            z_label="Train Accuracy", title=title + "-level CNN Classifier", path_config=path_config)
    plot_performance_metric(test_accuracies, learning_rates, weight_decays,
                            z_label="Validation Accuracy", title=title + "-level CNN Classifier", path_config=path_config)
    plot_performance_metric(train_losses, learning_rates, weight_decays, clip_outliers=True,
                            z_label="Train Loss", title=title + "-level CNN Classifier", path_config=path_config)
    plot_performance_metric(test_losses, learning_rates, weight_decays, clip_outliers=True,
                            z_label="Validation Loss", title=title + "-level CNN Classifier", path_config=path_config)


def plot_grid_search(path_config):
    LOGGER.info(
        "Using precomputed grid search results to plot all 3d-plots of the search space into the path_config['grid_search_results'] ...\n")
    experiment_logs = np.load(os.path.join(
        path_config["grid_search_path"], "grid_search_best_cnn_logs.npy"))

    models = segregate_models(experiment_logs)
    for level, model in zip(["Phylum", "Class", "Order"], models):
        plot_grid_search_plots(model, level, path_config)
