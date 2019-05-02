import ml.utils as utils
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import keras
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import argparse
import re
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['A', 'C', 'G', 'T', 'Z']))
RUN_OPTIONS = ["lstm_vae_ordinal", "lstm_vae_kmer_4", "lstm_vae_kmer_5"]
import ml.utils as utils
import warnings

def plotFigs3D(est, pred_new, savepath, args, predicted, path_config, scheme):
    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(predicted)
    labels = est.labels_
    for i in range(labels.shape[0]):
        ax.scatter(pred_new[i, 0], pred_new[i, 1], pred_new[i, 2],
                   c=scheme[labels[i]], edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(savepath)
    plt.savefig(path_config[args.model_name]['results'] + "/" + savepath)

def plotFigs2D(est, pred_new, savepath, args, predicted, path_config, scheme):
    plt.figure(figsize=(4, 3))
    est.fit(predicted)
    labels = est.labels_
    for i in range(labels.shape[0]):
        plt.scatter(pred_new[i, 0], pred_new[i, 1],
                   c=scheme[labels[i]], edgecolor='k')


    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(savepath)
    plt.savefig(path_config[args.model_name]['results'] + "/" + savepath)
