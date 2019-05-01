import os
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm

import ml.utils as utils


def get_kmer_dict(seq, k=3):
    bases = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in itertools.product(bases, repeat=k)]

    kmer_counts = {kmer: 0 for kmer in kmers}

    for kgram in [seq[i: i+k] for i in range(len(seq) - k + 1)]:
        kmer_counts[kgram] += 1

    return kmer_counts


def kmerize_data(fpath_csv, fpath_kmer_csv, k=3):
    df_data = pd.read_csv(fpath_csv)

    features = []
    fwrite_started = False
    for idx, row in tqdm(df_data.iterrows(), total=df_data.shape[0]):
        fdict = {'id': row['id']}
        kmers = get_kmer_dict(row['sequence'], k)

        maxval = max(list(kmers.values()))

        kmers_norm = {k: v/maxval for k, v in kmers.items()}

        fdict = {**fdict, **kmers_norm}
        if 'label' in row:
            fdict['label'] = row['label']
        if 'phylum' in row:
            fdict['phylum'] = row['phylum']
        if 'class' in row:
            fdict['class'] = row['class']
        if 'order' in row:
            fdict['order'] = row['order']

        features.append(fdict)

        if idx % 5000 == 0 and idx != 0:
            fwrite_started = True
            df_kmers = pd.DataFrame(features)
            mode = 'a' if idx != 5000 else 'w'
            header = (idx == 5000)
            df_kmers.to_csv(fpath_kmer_csv, index=None,
                            mode=mode, header=header)
            features = []

    if len(features) > 0:
        df_kmers = pd.DataFrame(features)
        mode = 'a' if fwrite_started else 'w'
        header = (not fwrite_started)
        df_kmers.to_csv(fpath_kmer_csv, index=None, mode=mode, header=header)


def generate_kmers(fpath_hierarchy, fpath_kmer):
    logger = utils.get_logger()
    for level in ['phylum', 'class', 'order']:
        logger.info('Generating K-Mers for {}'.format(level))

        fpath_level = os.path.join(fpath_kmer, level)

        if not os.path.exists(fpath_level):
            os.makedirs(fpath_level)

        for k in range(1, 7):
            logger.info('K={}'.format(k))
            kmerize_data(
                '{hie}/{level}/train.csv'.format(
                    hie=fpath_hierarchy, level=level),
                '{fpath}/train_{k}mer.csv'.format(
                    fpath=fpath_level, k=k
                ), k)

            kmerize_data(
                '{hie}/{level}/val.csv'.format(hie=fpath_hierarchy,
                                               level=level),
                '{fpath}/val_{k}mer.csv'.format(
                    fpath=fpath_level, level=level, k=k
                ), k)
