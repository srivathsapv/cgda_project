import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import ml.utils as utils

def train_kmer_for_level(model, level, k, df_train):
    trainX = df_train.values[:, :-2].astype(np.float16)
    trainY = df_train['label'].values.astype(np.int8)

    model.fit(trainX, trainY)

    predY = model.predict(trainX)
    f1 = f1_score(trainY, predY, average='macro')
    return float(f1)

def train_basic(models, dirpath_kmer, dirpath_output, kmin, kmax, verbose=True):
    logger = utils.get_logger(verbose)

    for model in models:
        model_str = type(model).__name__.lower()

        for level in ['phylum', 'class', 'order']:
            for k in range(kmin, kmax+1):
                df_train = pd.read_csv('{}/{}/train_{}mer.csv'.format(dirpath_kmer, level, k))
                f1_score = train_kmer_for_level(model, level, k, df_train)

                logger.info('Train F1 Score for {} model for {} level and k={} is {:.3f}'.format(model_str, level, k, f1_score))
