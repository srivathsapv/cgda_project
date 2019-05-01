from sklearn.preprocessing import LabelEncoder
import sys
import os
import argparse
import numpy as np
import pandas as pd

from scipy.stats import ttest_ind
from sklearn.svm import LinearSVC, SVC

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, f1_score, precision_recall_fscore_support, confusion_matrix

import h5py
import re
import ml.utils as utils
import warnings
warnings.filterwarnings("ignore")

label_encoder = LabelEncoder()
label_encoder.fit(np.array(['A', 'C', 'G', 'T', 'Z']))


def string_to_array(my_string):
    my_string = re.sub('[^ACGT]', 'Z', my_string)
    my_array = np.array(list(my_string))
    return my_array


def ordinal_encoder(my_array):
    integer_encoded = label_encoder.transform(my_array)
    float_encoded = integer_encoded.astype(float)
    data = []
    for i in range(float_encoded.shape[0]):
        if float_encoded[i] == 0:
            data.append([1, 0, 0, 0, 0])
        elif float_encoded[i] == 1:
            data.append([0, 1, 0, 0, 0])
        elif float_encoded[i] == 2:
            data.append([0, 0, 1, 0, 0])
        elif float_encoded[i] == 3:
            data.append([0, 0, 0, 1, 0])
        elif float_encoded[i] == 4:
            data.append([0, 0, 0, 0, 1])
    return data


def train_basic(dirpath_vector, dirpath_output):
    logger = utils.get_logger()
    x_train = np.genfromtxt(
        dirpath_vector + '/phylum/train.csv', delimiter='\n', dtype=None, encoding=None)
    x_test = np.genfromtxt(dirpath_vector + '/phylum/test.csv',
                           delimiter='\n', dtype=None, encoding=None)
    x_val = np.genfromtxt(dirpath_vector + '/phylum/val.csv',
                          delimiter='\n', dtype=None, encoding=None)
    arr = []
    arr1 = []
    arr2 = []

    for item in x_train[1:]:
        arr.append(ordinal_encoder(string_to_array(item.split(",")[3])))

    for item in x_test[1:]:
        arr1.append(ordinal_encoder(string_to_array(item.split(",")[3])))

    for item in x_val[1:]:
        arr2.append(ordinal_encoder(string_to_array(item.split(",")[3])))

    maxi = 0
    for item in arr:
        if len(item) > maxi:
            maxi = len(item)

    final1 = np.zeros((x_train.shape[0] - 1, maxi, 5))

    count = 0
    for item in arr:
        final1[count][:len(item)] = item
        count += 1

    maxi1 = 0
    for item in arr1:
        if len(item) > maxi1:
            maxi1 = len(item)

    final2 = np.zeros((x_test.shape[0] - 1, maxi1, 5))

    count = 0
    for item in arr1:
        final2[count][:len(item)] = item
        count += 1

    maxi2 = 0
    for item in arr2:
        if len(item) > maxi2:
            maxi2 = len(item)

    final3 = np.zeros((x_val.shape[0] - 1, maxi2, 5))

    count = 0
    for item in arr2:
        final3[count][:len(item)] = item
        count += 1

    hf = h5py.File(dirpath_vector + '/phylum/ordinal.h5', 'w')

    hf.create_dataset('dataset_1', data=final1)
    hf.create_dataset('dataset_2', data=final2)
    hf.create_dataset('dataset_3', data=final3)

    hf.close()

    x_train = np.genfromtxt(
        dirpath_vector + '/class/train.csv', delimiter='\n', dtype=None, encoding=None)
    x_test = np.genfromtxt(dirpath_vector + '/class/test.csv',
                           delimiter='\n', dtype=None, encoding=None)
    x_val = np.genfromtxt(dirpath_vector + '/class/val.csv',
                          delimiter='\n', dtype=None, encoding=None)
    arr = []
    arr1 = []
    arr2 = []

    for item in x_train[1:]:
        arr.append(ordinal_encoder(string_to_array(item.split(",")[3])))

    for item in x_test[1:]:
        arr1.append(ordinal_encoder(string_to_array(item.split(",")[3])))

    for item in x_val[1:]:
        arr2.append(ordinal_encoder(string_to_array(item.split(",")[3])))

    maxi = 0
    for item in arr:
        if len(item) > maxi:
            maxi = len(item)

    final1 = np.zeros((x_train.shape[0] - 1, maxi, 5))

    count = 0
    for item in arr:
        final1[count][:len(item)] = item
        count += 1

    maxi1 = 0
    for item in arr1:
        if len(item) > maxi1:
            maxi1 = len(item)

    final2 = np.zeros((x_test.shape[0] - 1, maxi1, 5))

    count = 0
    for item in arr1:
        final2[count][:len(item)] = item
        count += 1

    maxi2 = 0
    for item in arr2:
        if len(item) > maxi2:
            maxi2 = len(item)

    final3 = np.zeros((x_val.shape[0] - 1, maxi2, 5))

    count = 0
    for item in arr2:
        final3[count][:len(item)] = item
        count += 1

    hf = h5py.File(dirpath_vector + '/class/ordinal.h5', 'w')

    hf.create_dataset('dataset_1', data=final1)
    hf.create_dataset('dataset_2', data=final2)
    hf.create_dataset('dataset_3', data=final3)

    hf.close()

    hf = h5py.File(dirpath_vector + '/phylum/ordinal.h5', 'r')
    n1 = hf.get('dataset_1')
    n2 = hf.get('dataset_2')
    n3 = hf.get('dataset_3')
    X = np.array(n1)
    Y = np.array(n2)
    V = np.array(n3)
    hf.close()
    lab = np.genfromtxt(dirpath_vector + '/phylum/train.csv',
                        delimiter='\n', dtype=None, encoding=None)
    lab1 = np.genfromtxt(dirpath_vector + '/phylum/test.csv',
                         delimiter='\n', dtype=None, encoding=None)
    lab2 = np.genfromtxt(dirpath_vector + '/phylum/val.csv',
                         delimiter='\n', dtype=None, encoding=None)

    labels = []
    i = 0
    for item in lab[1:]:
        if item.split(",")[0][0] == "A":
            labels.append(0)
        elif item.split(",")[0][0] == "F":
            labels.append(1)
        else:
            labels.append(2)
        i += 1

    labels1 = []
    i = 0
    for item in lab1[1:]:
        if item.split(",")[0][0] == "A":
            labels1.append(0)
        elif item.split(",")[0][0] == "F":
            labels1.append(1)
        else:
            labels1.append(2)
        i += 1

    labels2 = []
    i = 0
    for item in lab2[1:]:
        if item.split(",")[0][0] == "A":
            labels2.append(0)
        elif item.split(",")[0][0] == "F":
            labels2.append(1)
        else:
            labels2.append(2)
        i += 1

    label = np.array(labels)
    label1 = np.array(labels1)
    label2 = np.array(labels2)

    clf2 = SVC(kernel='rbf')
    clf = RandomForestClassifier()

    newX = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    newY = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])
    clf2.fit(newX, label)
    clf.fit(newX, label)

    preds2 = clf2.predict(newX)
    preds = clf.predict(newX)

    preds2_test = clf2.predict(newY)
    preds_test = clf.predict(newY)
    np.save(dirpath_output + '/SVM_phylum_predictions', preds2_test)
    np.save(dirpath_output + '/RF_phylum_predictions', preds_test)

    scores = clf2.decision_function(newX)
    scores2 = clf.predict(newX)

    score = np.amax(scores, axis=1)

    fpr, tpr, thresholds = roc_curve(label, score, pos_label=2)
    fpr2, tpr2, thresholds2 = roc_curve(label, scores2, pos_label=2)

    match2 = 0
    for i in range(preds2.shape[0]):
        if preds2[i] == label[i]:
            match2 += 1
    accuracy2 = float(match2) / preds2.shape[0]
    p, r, f1, s = precision_recall_fscore_support(
        label, preds2, average='weighted')

    match = 0
    for i in range(preds.shape[0]):
        if preds[i] == label[i]:
            match += 1
    accuracy = float(match) / preds.shape[0]
    p2, r2, f12, s = precision_recall_fscore_support(
        label, preds, average='weighted')

    C = confusion_matrix(label, preds2)

    logger.info('Train Accuracy, precision, recall and F1 Score for SVM model for phylum level is {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        accuracy2, p, r, f1))
    logger.info('Train Accuracy, precision, recall and F1 Score for Random Forest model for phylum level is {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        accuracy, p2, r2, f12))

    hf = h5py.File(dirpath_vector + '/class/ordinal.h5', 'r')
    n1 = hf.get('dataset_1')
    n2 = hf.get('dataset_2')
    n3 = hf.get('dataset_3')
    X = np.array(n1)
    Y = np.array(n2)
    V = np.array(n3)
    hf.close()

    lab = np.genfromtxt(dirpath_vector + '/class/train.csv',
                        delimiter='\n', dtype=None, encoding=None)
    lab1 = np.genfromtxt(dirpath_vector + '/class/test.csv',
                         delimiter='\n', dtype=None, encoding=None)
    lab2 = np.genfromtxt(dirpath_vector + '/class/val.csv',
                         delimiter='\n', dtype=None, encoding=None)

    labels = []
    i = 0
    for item in lab[1:]:
        labels.append(int(item.split(",")[2]))
        i += 1

    labels1 = []
    i = 0
    for item in lab1[1:]:
        labels1.append(int(item.split(",")[2]))
        i += 1

    labels2 = []
    i = 0
    for item in lab2[1:]:
        labels2.append(int(item.split(",")[2]))
        i += 1

    label = np.array(labels)
    label1 = np.array(labels1)
    label2 = np.array(labels2)

    clf2 = RandomForestClassifier()
    clf = SVC(kernel='rbf')

    newX = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    newY = Y.reshape(Y.shape[0], Y.shape[1] * Y.shape[2])
    clf2.fit(newX, label)
    clf.fit(newX, label)
    preds2 = clf2.predict(newX)
    preds = clf.predict(newX)
    scores = clf2.predict(newX)
    scores1 = clf.decision_function(newX)

    preds2_test = clf2.predict(newY)
    preds_test = clf.predict(newY)

    np.save(dirpath_output + '/SVM_class_predictions', preds2_test)
    np.save(dirpath_output + '/RF_class_predictions', preds_test)

    score = np.amax(scores1, axis=1)

    fpr, tpr, thresholds = roc_curve(label, scores, pos_label=2)
    fpr2, tpr2, thresholds2 = roc_curve(label, score, pos_label=2)

    match2 = 0
    for i in range(preds2.shape[0]):
        if preds2[i] == label[i]:
            match2 += 1
    accuracy2 = float(match2) / preds2.shape[0]
    p, r, f1, s = precision_recall_fscore_support(
        label, preds2, average='weighted')
    C = confusion_matrix(label, preds2)

    match = 0
    for i in range(preds.shape[0]):
        if preds[i] == label[i]:
            match += 1
    accuracy = float(match) / preds.shape[0]
    p2, r2, f12, s = precision_recall_fscore_support(
        label, preds, average='weighted')

    logger.info('Train Accuracy, precision, recall and F1 Score for SVM model for phylum level is {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        accuracy, p2, r2, f12))
    logger.info('Train Accuracy, precision, recall and F1 Score for Random Forest model for phylum level is {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        accuracy2, p, r, f1))
