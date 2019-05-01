import os
import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

plt.style.use('seaborn')

warnings.filterwarnings("ignore")


def plot_train_eval_curves(acc_train, acc_test, loss_train, loss_test, save_path):
    print("\n")
    print("Train Accuracy:", str(acc_train[-1]))
    print("Train Loss:", str(loss_train[-1]))
    print("Test Accuracy:", str(acc_test[-1]))
    print("Test Loss:", str(loss_test[-1]))
    print("\n")

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
