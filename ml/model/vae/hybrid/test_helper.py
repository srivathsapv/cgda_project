import os
import numpy as np
import torch as t
import pandas as pd

from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import ml.utils as utils
from ml.model.vae.hybrid.batchloader import BatchLoader
from ml.model.vae.hybrid.parameters import Parameters
from ml.model.vae.hybrid.vae import VAE

COLORS = ['tab:blue', 'tab:brown', 'tab:green', 'tab:red', 'tab:purple',
          'tab:orange', 'tab:pink', 'black', 'tab:olive', 'tab:cyan']

def test_vae(path_config, feature_type, hyperparams, model_name):
    is_kmer = ('kmer' in feature_type)
    fpath_data = (path_config['features']
                  if is_kmer else path_config['sequences'])

    fpath_data = os.path.dirname(os.path.abspath(fpath_data))
    dirpath_results = path_config['results']
    use_gpu = t.cuda.is_available()

    save_loss_curve(dirpath_results, model_name)

    embeddings = get_embeddings(
        fpath_data, is_kmer, feature_type, use_gpu, dirpath_results, model_name
    )

    save_embed_plots(fpath_data, embeddings, dirpath_results, model_name)

def get_embeddings(fpath_data, is_kmer, feature_type, use_gpu, dirpath_results,
                    model_name):

    logger = utils.get_logger()
    logger.info('Running inference and obtaining embeddings for {}'.format(feature_type))

    if is_kmer:
        kval = int(feature_type.split('_')[1])
        fpath_features = '{}/features_{}mer.npy'.format(fpath_data, kval)
        features = np.load(fpath_features)
    else:
        fpath_features = '{}/ordinal_sequences.txt'.format(fpath_data)
        features = open(fpath_features, 'r').read().split('\n')

    batch_loader = BatchLoader(data_path=fpath_features, is_kmer=(is_kmer))
    parameters = Parameters(batch_loader.vocab_size, feature_type=feature_type)

    vae = VAE(parameters)

    if use_gpu:
        vae = vae.cuda()

    vae.load_state_dict(t.load('{}/{}_best.pth'.format(dirpath_results, model_name)))

    batch_size = 5000
    batch_indices = [i for i in np.arange(start=batch_size, stop=len(features), step=batch_size)]
    feature_batches = np.split(features, batch_indices)

    embeddings = []

    for feature_batch in feature_batches:
        if not is_kmer:
            idxs = [batch_loader._get_idxs(seq) for seq in feature_batch]
            encoder_input, _, _ = batch_loader._wrap_tensor(idxs, use_cuda=use_gpu)
        else:
            encoder_input, _, _ = batch_loader._wrap_tensor(feature_batch, use_cuda=use_gpu)
        mu, _ = vae.inference(encoder_input)
        embeddings.extend(mu.data.cpu().numpy())

    embeddings = np.array(embeddings)
    return embeddings

def save_loss_curve(dirpath_results, model_name):
    logger = utils.get_logger()
    df_metrics = pd.read_csv('{}/{}_metrics.csv'.format(dirpath_results, model_name), index_col=0)
    fpath_plot = '{}/{}_loss.png'.format(dirpath_results, model_name)

    train_ces = df_metrics['train_ce'].values
    valid_ces = df_metrics['valid_ce'].values

    if 'kmer' in model_name:
        train_ces = np.clip(train_ces, a_min=min(train_ces), a_max=0.07)
        valid_ces = np.clip(valid_ces, a_min=min(valid_ces), a_max=0.07)

    iterations = [i * 50 for i in range(df_metrics.shape[0])]

    plt.plot(iterations, train_ces, color=COLORS[0], label='Train CE')
    plt.plot(iterations, valid_ces, color=COLORS[1], label='Validation CE')

    plt.xlabel('Iteration')
    plt.ylabel('Cross Entropy')
    plt.title('Cross entropy curves {}'.format(model_name), fontsize=16)
    plt.legend()
    plt.savefig(fpath_plot)
    plt.clf()

    logger.info('Saved loss curves in {}'.format(fpath_plot))


def save_embed_plots(fpath_data, embeddings, dirpath_results, model_name):
    df_data = pd.read_csv('data/embeds.csv')

    embed_configs = [
        {
            'model': PCA(n_components=10, whiten=True),
            'fpath_plot': '{}/{}_pca_plot.png'.format(dirpath_results, model_name),
            'xlabel': 'PC 1',
            'ylabel': 'PC 2',
            'title': 'PCA of phylum embeddings for {} model'.format(model_name),
            'info_msg': 'Saved PCA plot in {path}'
        },
        {
            'model': TSNE(n_components=3, n_jobs=8, verbose=1, n_iter=250),
            'fpath_plot': '{}/{}_tsne_plot.png'.format(dirpath_results, model_name),
            'xlabel': 't-SNE 1',
            'ylabel': 't-SNE 2',
            'title': 'TCA of phylum embeddings for {} model'.format(model_name),
            'info_msg': 'Saved TSNE plot in {path}'
        }
    ]

    for embed_config in embed_configs:
        embed_and_plot(df_data, embeddings, embed_config)


def embed_and_plot(df_data, embeddings, embed_config):
    logger = utils.get_logger()
    embed_data = embed_config['model'].fit_transform(embeddings)

    phy_labels = sorted(list(df_data['phylum'].value_counts().index[:3]))
    phy_indices = [i for i, _ in enumerate(phy_labels)]

    fpath_plot = embed_config['fpath_plot']

    for phy_idx in phy_indices:
        data_indices = np.where(df_data['phylum'].values == phy_labels[phy_idx])
        data_pts = embed_data[data_indices]
        plt.scatter(
            data_pts[:, 0], data_pts[:, 1], color=COLORS[phy_idx],
            label=phy_labels[phy_idx]
        )

    plt.xlabel(embed_config['xlabel'])
    plt.ylabel(embed_config['xlabel'])
    plt.title(embed_config['title'], fontsize=16)
    plt.legend()
    plt.savefig(fpath_plot)
    plt.clf()

    logger.info(embed_config['info_msg'].format(path=fpath_plot))
