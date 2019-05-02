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


from ml.model.vae.lstm.network import create_lstm_vae
from ml.model.vae.lstm.plotter import plotFigs2D,plotFigs3D
from ml.model.vae.lstm.extract_data import get_data1,get_data2
from ml.model.vae.lstm.encode_helper import ordinal_encoder, string_to_array
warnings.filterwarnings("ignore")


def train_model(path_config, args=None):
    x = get_data1(path_config[args.model_name]['train'], args)
    x_test = get_data2(path_config[args.model_name]['test'], args)
    hyperparams = utils.get_model_hyperparams('lstm_vae')
    dirpath_results = path_config[args.model_name]['results']
    logger = utils.get_logger()
    logger.info((x.shape, x_test.shape))
    input_dim = x.shape[-1]  # 13
    timesteps = x.shape[1]  # 3
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
    logger.info(predicted.shape)


def test_model(path_config, args=None):
    logger = utils.get_logger()
    predicted = np.load(path_config[args.model_name]['results']+ "/" + args.model_name + "_predicted.npy")
    lab = np.genfromtxt(path_config[args.model_name]['test'],delimiter='\n',dtype=None,encoding=None)
    labels = []
    scheme = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    if args.model_name == "lstm_vae_ordinal":
      i = 0
      for item in lab[1:]:
        if item.split(",")[0][0] == "P":
          labels.append(0)
        elif item.split(",")[0][0] == "F":
          labels.append(1)
        else:
          labels.append(2)
        i+=1
    else:
      i = 0
      for item in lab[1:]:
        labels.append(float(item.split(",")[-1]))
        i+=1

    label = np.array(labels)

    clf = PCA(n_components=3)
    pred_new = clf.fit_transform(predicted)

    est = KMeans(n_clusters=10, init = predicted[:10])

    plotFigs3D(est, pred_new, 'hclustering_order_10-3D', args, predicted, path_config, scheme)

    est2 = KMeans(n_clusters=5)

    clf2 = PCA(n_components=3)

    pred_new2 = clf2.fit_transform(est.cluster_centers_)

    plotFigs3D(est2, pred_new2, 'hclustering_class_5-3D', args, est.cluster_centers_, path_config, scheme)


    est3 = KMeans(n_clusters=3)

    clf3 = PCA(n_components=3)
    pred_new3 = clf3.fit_transform(est2.cluster_centers_)

    plotFigs3D(est3, pred_new3, 'hclustering_phylum_3-3D', args, est2.cluster_centers_, path_config, scheme)


    fig = plt.figure(figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    labelslis = []
    for i in range(label.shape[0]):
        labelslis.append(scheme[label[i]])

    ax.scatter(pred_new[:, 0], pred_new[:, 1], pred_new[:, 2],
               c=labelslis, edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D_groundClustering')
    plt.savefig(path_config[args.model_name]['results'] + '/3D_groundClustering')

    logger.info('3D clustering completed, plot stored in ' + path_config[args.model_name]['results'])

    # from mpl_toolkits.mplot3d import Axes3D
    clf = PCA(n_components=2)
    pred_new = clf.fit_transform(predicted)

    est = KMeans(n_clusters=10,init = predicted[:10])

    plotFigs2D(est, pred_new, 'hclustering_order_10-2D', args, predicted, path_config, scheme)

    clf2 = PCA(n_components=2)
    pred_new2 = clf2.fit_transform(est.cluster_centers_)

    est2 = KMeans(n_clusters=5)

    plotFigs2D(est2, pred_new2, 'hclustering_class_5-2D', args, est.cluster_centers_, path_config, scheme)

    clf3 = PCA(n_components=2)
    pred_new3 = clf3.fit_transform(est2.cluster_centers_)

    est3 = KMeans(n_clusters=3)

    plotFigs2D(est3, pred_new3, 'hclustering_phylum_3-2D', args, est2.cluster_centers_, path_config, scheme)

    clf_ground = PCA(n_components=2)
    pred_new_ground = clf_ground.fit_transform(predicted)

    plt.figure(figsize=(4, 3))
    labelslis = []
    for i in range(label.shape[0]):
        labelslis.append(scheme[label[i]])

    plt.scatter(pred_new_ground[:, 0], pred_new_ground[:, 1],
               c=labelslis, edgecolor='k')


    plt.xlabel('PC1')
    plt.ylabel('PC2')
    # ax.set_zlabel('Petal length')
    plt.title('hclustering_ground2D')
    plt.savefig(path_config[args.model_name]['results'] + '/hclustering_ground2D')
    logger.info('2D clustering completed, plot stored in ' + path_config[args.model_name]['results'])
