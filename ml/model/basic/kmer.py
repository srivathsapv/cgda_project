import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import ml.utils as utils

def train_kmer_for_level(model, level, k, df_train, df_val):
    trainX = df_train.values[:, :-2].astype(np.float16)
    trainY = df_train['label'].values.astype(np.int8)

    valX = df_val.values[:, :-2].astype(np.float16)
    valY = df_val['label'].values.astype(np.int8)

    model.fit(trainX, trainY)

    predY = model.predict(trainX)
    train_f1 = f1_score(trainY, predY, average='macro')

    pred_val = model.predict(valX)
    return float(train_f1), pred_val

def train_kmer_combined(model, level, train_data, train_labels, val_data, val_labels):
    model.fit(train_data, train_labels)
    predY = model.predict(train_data)
    f1 = f1_score(train_labels, predY, average='macro')

    pred_val = model.predict(val_data)

    return float(f1), pred_val

def train_basic(models, dirpath_kmer, dirpath_output, kmin, kmax, verbose=True):
    logger = utils.get_logger(verbose)

    for model in models:
        model_str = type(model).__name__.lower()

        for level in ['phylum', 'class', 'order']:
            combined_train_data = []
            combined_val_data = []

            for k in range(kmin, kmax+1):
                df_train = pd.read_csv('{}/{}/train_{}mer.csv'.format(dirpath_kmer, level, k))
                df_val = pd.read_csv('{}/{}/val_{}mer.csv'.format(dirpath_kmer, level, k))

                train_f1, pred_val = train_kmer_for_level(model, level, k, df_train, df_val)

                combined_train_data.append(df_train.values[:, :-2].astype(np.float16))
                combined_val_data.append(df_val.values[:, :-2].astype(np.float16))

                logger.info('Train F1 Score for {} model for {} level and k={} is {:.3f}'.format(model_str, level, k, train_f1))
                np.save('{}/preds_{}_{}mer.npy'.format(dirpath_output, level, k), pred_val)

            combined_train_data = np.hstack(combined_train_data)
            combined_val_data = np.hstack(combined_val_data)

            train_labels = df_train['label'].values.astype(np.int8)
            val_labels = df_val['label'].values.astype(np.int8)

            combined_f1, combined_pred = train_kmer_combined(
                model, level, combined_train_data, train_labels,
                combined_val_data, val_labels
            )

            logger.info(('Train F1 Score for {} model for {} level and ' +
                        'combined K from 1-5 is {:.3f}').format(model_str, level, combined_f1))
            np.save('{}/preds_{}_combined.npy'.format(dirpath_output, level), combined_pred)
