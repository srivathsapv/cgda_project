import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
plt.style.use('seaborn')

import ml.utils as utils

COLORS = ['tab:blue', 'tab:brown', 'tab:green', 'tab:red', 'tab:purple',
          'tab:orange', 'tab:pink', 'black', 'tab:olive', 'tab:cyan']

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


def train_kmer_combined(model, level, train_data,
                        train_labels, val_data, val_labels):
    model.fit(train_data, train_labels)
    predY = model.predict(train_data)
    f1 = f1_score(train_labels, predY, average='macro')

    pred_val = model.predict(val_data)

    return float(f1), pred_val


def train_basic(models, dirpath_kmer, dirpath_output, kmin, kmax):
    logger = utils.get_logger()

    for model in models:
        model_str = type(model).__name__.lower()

        for level in ['phylum', 'class', 'order']:
            combined_train_data = []
            combined_val_data = []

            for k in range(kmin, kmax + 1):
                df_train = pd.read_csv(
                    '{}/{}/train_{}mer.csv'.format(dirpath_kmer, level, k))
                df_val = pd.read_csv(
                    '{}/{}/val_{}mer.csv'.format(dirpath_kmer, level, k))

                train_f1, pred_val = train_kmer_for_level(
                    model, level, k, df_train, df_val)

                combined_train_data.append(
                    df_train.values[:, :-2].astype(np.float16))
                combined_val_data.append(
                    df_val.values[:, :-2].astype(np.float16))

                logger.info('Train F1 Score for {} model for {} level and k={} is {:.3f}'.format(
                    model_str, level, k, train_f1))
                np.save('{}/{}_preds_{}_{}mer.npy'.format(dirpath_output,
                                                          model_str, level, k), pred_val)

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
            np.save('{}/{}_preds_{}_combined.npy'.format(dirpath_output,
                                                         model_str, level), combined_pred)

def plot_kmer_metrics(path_config, args):
    logger = utils.get_logger()

    path_config = path_config['basic_kmer']

    dirpath_kmer = path_config['dirpath_kmer']
    dirpath_results = path_config['results']

    model_display = {
        'svc': 'SVM',
        'randomforestclassifier': 'Random Forest'
    }

    xticks = [str(i) for i in range(1, 7)]
    xticks.append('combined')

    for model in ['svc', 'randomforestclassifier']:
        model_scores = {}
        for i, level in enumerate(['phylum', 'class', 'order']):
            model_scores[level] = []
            df_data = pd.read_csv('{}/{}/val_1mer.csv'.format(dirpath_kmer, level))
            gt_y = df_data['label'].values.astype(np.int8)

            for k in range(1, 7):
                pred_y = np.load('{}/{}_preds_{}_{}mer.npy'.format(dirpath_results, model, level, k))
                model_scores[level].append(f1_score(gt_y, pred_y, average='macro'))
            pred_y = np.load('{}/{}_preds_{}_combined.npy'.format(dirpath_results, model, level))
            model_scores[level].append(f1_score(gt_y, pred_y, average='macro'))

            plt.plot(range(1, 8), model_scores[level], label=level, color=COLORS[i])

        fpath_plot = '{}/{}_kmer.png'.format(dirpath_results, model)
        plt.xticks(range(1, 8), xticks)
        plt.xlabel('K-Mer length')
        plt.ylabel('F1 Score')
        plt.title('F1 Metrics for {}'.format(model_display[model]))
        plt.legend()
        plt.savefig(fpath_plot)
        logger.info('Saving K-Mer comparison plot for {} in {}'.format(model_display[model], fpath_plot))
        plt.clf()
