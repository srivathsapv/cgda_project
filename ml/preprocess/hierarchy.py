import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn import preprocessing

import ml.utils as utils

MAX_SAMPLES_PER_CLASS = 10000
ORDER_OTHER_LABELS = [
    'Coriobacteriales', 'Rhodocyclales', 'Xanthomonadales', 'Alteromonadales',
    'Caulobacterales', 'Rhodobacterales', 'Rhodospirillales', 'Vibrionales',
    'Oceanospirillales', 'Acidimicrobiales'
]


def create_split_csv(data, labels, fpath_split):
    label_encoder = preprocessing.LabelEncoder()
    label_nos = label_encoder.fit_transform(labels)
    data_rows = []
    for i, (row_id, sequence) in enumerate(data):
        data_dict = {}
        data_dict['id'] = row_id
        data_dict['sequence'] = sequence
        data_dict['class_name'] = labels[i]
        data_dict['label'] = label_nos[i]
        data_rows.append(data_dict)

    df_split = pd.DataFrame(data_rows)
    df_split.to_csv(fpath_split, index=False)


def get_xy_from_df(df_taxa, level):
    ids = df_taxa['id'].values
    seqs = df_taxa['sequence'].values

    X = np.vstack([ids, seqs]).T
    y = df_taxa[level].values

    return X, y


def split_data(df_taxa, level, n_samples_per_class, fpath_hierarchy):
    logger = utils.get_logger()
    fpath_level = os.path.join(fpath_hierarchy, level)

    if not os.path.exists(fpath_level):
        os.makedirs(fpath_level)

    logger.info('Creating hierarchy data for {} level'.format(level))
    df_group = df_taxa.sample(frac=1).groupby(by=level)

    X = []
    y = []
    for name, group in df_group:
        gX, gy = get_xy_from_df(group, level)
        gX = gX[:int(n_samples_per_class)]
        gy = gy[:int(n_samples_per_class)]
        X.append(gX)
        y.append(gy)

    X = np.vstack(X)
    y = np.concatenate(y)

    (train_data, testval_data, train_labels, testval_labels) = \
        train_test_split(X, y, test_size=0.4, stratify=y, shuffle=True)

    (test_data, val_data, test_labels, val_labels) = \
        train_test_split(testval_data, testval_labels,
                         test_size=0.5, stratify=testval_labels, shuffle=True)

    create_split_csv(train_data, train_labels,
                     '{}/train.csv'.format(fpath_level, level))
    create_split_csv(test_data, test_labels,
                     '{}/test.csv'.format(fpath_level, level))
    create_split_csv(val_data, val_labels,
                     '{}/val.csv'.format(fpath_level, level))


def group_labels(df_taxa, label_names):
    df_group = df_taxa
    grouped_label_name = 'Other'

    for lname in label_names:
        df_group = df_group.replace(lname, grouped_label_name)
    return df_group


def create_hierarchy(fpath_taxa, fpath_hierarchy):
    logger = utils.get_logger()
    logger.info('Creating hierarchy files from taxa.csv')
    if not os.path.exists(fpath_hierarchy):
        os.makedirs(fpath_hierarchy)

    df_taxa = pd.read_csv(fpath_taxa)

    split_data(df_taxa, 'phylum', MAX_SAMPLES_PER_CLASS, fpath_hierarchy)
    split_data(df_taxa, 'class', MAX_SAMPLES_PER_CLASS, fpath_hierarchy)

    df_group = group_labels(df_taxa, ORDER_OTHER_LABELS)
    split_data(df_group, 'order', MAX_SAMPLES_PER_CLASS, fpath_hierarchy)
