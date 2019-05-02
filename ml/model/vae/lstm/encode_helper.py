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

def string_to_array(my_string):
    my_string = re.sub('[^ACGT]', 'Z', my_string)
    my_array = np.array(list(my_string))
    return my_array


def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 0.25  # A
    float_encoded[float_encoded == 1] = 0.50  # C
    float_encoded[float_encoded == 2] = 0.75  # G
    float_encoded[float_encoded == 3] = 1.00  # T
    float_encoded[float_encoded == 4] = 1.25
    return float_encoded