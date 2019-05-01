import os
import numpy as np
import pandas as pd

from ml.preprocess.kmer import kmerize_data

import ml.utils as utils


def generate_kmers(fpath_embeds, dirpath_vae, k):
    logger = utils.get_logger()
    logger.info('Generating {}-mers for VAE'.format(k))

    fpath_embed_kmer = '{}/embeds_{}mer.csv'.format(dirpath_vae, k)
    kmerize_data(fpath_embeds, fpath_embed_kmer, k)

    df_kmer = pd.read_csv(fpath_embed_kmer).drop(
        ['id', 'phylum', 'class', 'order'], axis=1)
    data_kmer = df_kmer.values.astype(np.float16)
    np.save('{}/features_{}mer.npy'.format(dirpath_vae, k), data_kmer)


def generate_vae_data(fpath_embeds, dirpath_vae):
    logger = utils.get_logger()
    logger.info('Generating data files for VAE training')
    if not os.path.exists(dirpath_vae):
        os.makedirs(dirpath_vae)

    df_embeds = pd.read_csv(fpath_embeds)
    seqs = df_embeds['sequence'].values

    with open('{}/ordinal_sequences.txt'.format(dirpath_vae), 'w') as seqfile:
        seqfile.write('\n'.join(seqs))

    generate_kmers(fpath_embeds, dirpath_vae, 4)
    generate_kmers(fpath_embeds, dirpath_vae, 5)
