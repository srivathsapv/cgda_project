from ml.model.vae.lstm.encode_helper import ordinal_encoder, string_to_array
import warnings
from sklearn.preprocessing import LabelEncoder
import re
import argparse
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
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['A', 'C', 'G', 'T', 'Z']))
RUN_OPTIONS = ["lstm_vae_ordinal", "lstm_vae_kmer_4", "lstm_vae_kmer_5"]


def get_data1(path, args):
    # read data from file
    x_train = np.transpose(np.genfromtxt(
        path, delimiter='\n', dtype=None, encoding=None))
# x_test =
# np.transpose(np.genfromtxt(,delimiter='\n',dtype=None,encoding=None))
    if args.model_name == "lstm_vae_ordinal":
        arr = []
        for item in x_train[1:]:
            arr.append(ordinal_encoder(string_to_array(item.split(",")[9])))
    else:
        arr = []
        for item in x_train[1:]:
            arr.append(item.split(',')[:-4])
#       arr.append(item.split(",")[3])

    maxi = 0
    for item in arr:
        if len(item) > maxi:
            maxi = len(item)

    data = np.empty((x_train.shape[0] - 1, maxi))
    count = 0
    for item in arr:
        data[count][:len(item)] = item
        count += 1

    timesteps = 1
    dataX = []
    for i in range(data.shape[0]):
        x = data[i:(i + timesteps), :]
        dataX.append(x)
    return np.array(dataX)


def get_data2(path, args):
    x_train = np.transpose(np.genfromtxt(
        path, delimiter='\n', dtype=None, encoding=None))

    if args.model_name == "lstm_vae_ordinal":
        arr = []
        for item in x_train[1:]:
            arr.append(ordinal_encoder(string_to_array(item.split(",")[3])))
    else:
        arr = []
        for item in x_train[1:]:
            arr.append(item.split(',')[:-2])

    maxi = 0
    for item in arr:
        if len(item) > maxi:
            maxi = len(item)

    data = np.empty((x_train.shape[0] - 1, maxi))
    count = 0
    for item in arr:
        data[count][:len(item)] = item
        count += 1

    timesteps = 1
    dataX = []
    for i in range(data.shape[0]):
        x = data[i:(i + timesteps), :]
        dataX.append(x)
    return np.array(dataX)
