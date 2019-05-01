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
import argparse
import re
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(np.array(['A','C','G','T','Z']))
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ml.utils as utils
RUN_OPTIONS = ["lstm_vae_ordinal", "lstm_vae_kmer_4", "lstm_vae_kmer_5"]

def create_lstm_vae(input_dim,
    timesteps,
    batch_size,
    intermediate_dim,
    latent_dim,
    epsilon_std=1.):

    x = Input(shape=(timesteps, input_dim,))

    # LSTM encoding
    h = LSTM(intermediate_dim)(x)

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_sigma/2.0) * epsilon

    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = kl_loss + xent_loss
        return loss

    vae.compile(optimizer='rmsprop', loss=vae_loss)

    return vae, encoder, generator

def string_to_array(my_string):
    my_string = re.sub('[^ACGT]', 'Z', my_string)
    my_array = np.array(list(my_string))
    return my_array

def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    float_encoded[float_encoded == 0] = 0.25 # A
    float_encoded[float_encoded == 1] = 0.50 # C
    float_encoded[float_encoded == 2] = 0.75 # G
    float_encoded[float_encoded == 3] = 1.00 # T
    float_encoded[float_encoded == 4] = 1.25
    return float_encoded

def get_data1(path, args):
    # read data from file
    x_train = np.transpose(np.genfromtxt(path,delimiter='\n',dtype=None,encoding=None))
#     x_test = np.transpose(np.genfromtxt(,delimiter='\n',dtype=None,encoding=None))
    if args.model_name=="lstm_vae_ordinal":
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
      if len(item)>maxi:
        maxi = len(item)

    data = np.empty((x_train.shape[0]-1, maxi))
    count = 0
    for item in arr:
      data[count][:len(item)] = item
      count+=1

    timesteps = 1
    dataX = []
    for i in range(data.shape[0]):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)


def get_data2(path, args):
    x_train = np.transpose(np.genfromtxt(path,delimiter='\n',dtype=None,encoding=None))

    if args.model_name=="lstm_vae_ordinal":
        arr = []
        for item in x_train[1:]:
          arr.append(ordinal_encoder(string_to_array(item.split(",")[3])))
    else:
        arr = []
        for item in x_train[1:]:
          arr.append(item.split(',')[:-2])

    maxi = 0
    for item in arr:
      if len(item)>maxi:
        maxi = len(item)

    data = np.empty((x_train.shape[0]-1, maxi))
    count = 0
    for item in arr:
      data[count][:len(item)] = item
      count+=1

    timesteps = 1
    dataX = []
    for i in range(data.shape[0]):
        x = data[i:(i+timesteps), :]
        dataX.append(x)
    return np.array(dataX)

def train_model(path_config, args=None):
    x = get_data1(path_config[args.model_name]['train'], args)
    x_test = get_data2(path_config[args.model_name]['test'], args)
    hyperparams = utils.get_model_hyperparams('lstm_vae')
    dirpath_results = path_config[args.model_name]['results']
    logger = utils.get_logger()
    logger.info((x.shape, x_test.shape))
    input_dim = x.shape[-1] # 13
    timesteps = x.shape[1] # 3
    # batch_size = 1

    vae, enc, gen = create_lstm_vae(input_dim,
        timesteps,
        hyperparams['batch_size'],
        hyperparams['intermediate_dim'],
        hyperparams['latent_dim'],
        hyperparams['epsilon_std'])

    ep = hyperparams["num_iterations"]
    if args.is_demo:
        ep = 1

    vae.fit(x, x, epochs=ep)
    vae.save_weights(dirpath_results + args.model_name + "_weights.h5")
    logger.info(vae.summary())
#     preds = vae.predict(x, batch_size=batch_size)
    predicted = enc.predict(x_test, batch_size=hyperparams['batch_size'])
    np.save(dirpath_results + args.model_name + "_predicted", predicted)
    logger.info (predicted.shape)
