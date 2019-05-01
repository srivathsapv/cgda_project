import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
import datetime
import time
import pandas as pd
import glob
import os
import numpy as np

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt


def evaluate(test_data, model, dirpath_results, parameters, device, logger, exp_name):


    logdir = dirpath_results
    exp_dir = logdir + '/' + exp_name
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    state_files = glob.glob(exp_dir + '/*')
    for sf in state_files:
        sf_base = os.path.basename(sf)
        if sf_base == 'bestmodel.pt':
            load_state_file = sf

    if load_state_file is not None:
        state = torch.load(load_state_file, map_location=device)
        logger.info('Loaded saved model '+load_state_file)

    if state is not None:
        model.load_state_dict(state['enc_state'])

    acc,label_weights = inference(test_data, model, device)
    logger.info('Test accuracy: {}'.format(acc))

    
    logger.info('Plotting Accuracy vs. Epoch plot...')
    metrics_file = exp_dir + '/metrics_best.tsv'
    save_filename = exp_dir + '/' + exp_name +'_lineplot.png'
    with open(metrics_file) as fin:
        X = ([x.strip().split('\t') for x in fin.readlines()])

    trainacc=[]
    testacc=[]
    for i in range(len(X)):
        trainacc.append(X[i][0])
        testacc.append(X[i][2])
    
    trainacc = np.array(trainacc, dtype=np.float)
    testacc = np.array(testacc, dtype=np.float)
    if exp_name == 'rnn_phylum':
        title = 'Phylum LSTM Classifier'
    elif exp_name == 'rnn_class':
        title = 'Class LSTM Classifier'
    else:
        title = 'Order LSTM Classifier'

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax1.plot(range(len(trainacc)), trainacc, color='olivedrab', label='Train data')
    ax1.plot(range(len(testacc)), testacc, color='palevioletred', label='Test data')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy')
    ax1.set_title(title)
    ax1.legend()
    plt.savefig(save_filename)
    logger.info('Accuracy vs. Epoch plot can be found at '+save_filename)

    
    if exp_name == 'rnn_phylum':
        logger.info('Since Phylum level has only 3 labels, creating scatter plot to visualize the encoded outputs along the three dimensions...')
        test_labels=[]
        for data in test_data:
            test_labels.append(data[0])
        test_labels=np.array(test_labels, dtype=np.int)

        scatterplot_filename = exp_dir + '/rnn_phylum_scatterplot.png'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30, 30)
        colorlist = ['mediumvioletred', 'teal', '#0082c8']
        for i in range(len(label_weights)):
            if test_labels[i]==0:
                c1 = ax.scatter(label_weights[i, 0], label_weights[i, 1], label_weights[i, 2], color=colorlist[test_labels[i]], marker='*')
            elif test_labels[i]==1:
                c2 = ax.scatter(label_weights[i, 0], label_weights[i, 1], label_weights[i, 2], color=colorlist[test_labels[i]], marker='*')
            else:
                c3 = ax.scatter(label_weights[i, 0], label_weights[i, 1], label_weights[i, 2], color=colorlist[test_labels[i]], marker='*')
        ax.legend((c1, c2, c3), ('Actinobacteria', 'Firmicutes', 'Proteobacteria'), loc=0)
        ax.set_xlabel('Encoded component 1')   
        ax.set_ylabel('Encoded component 2')
        ax.set_zlabel('Encoded component 3')
        plt.savefig(scatterplot_filename)
        logger.info('Scatter plot of LSTM encoded components for Phylum Classifier can be found at '+scatterplot_filename)
    
    
    

    

def inference(inf_data, model, device):

    total_loss=0
    correct=0
    incorrect=0
    truth=[]
    preds=[]
    label_weights=[]
    for i, data in enumerate(inf_data):
        seq = data[1]
        label = data[0]
        ip = prep_single_seq(seq, device)
        gold = prep_single_label(label, device)
        with torch.no_grad():
            model.hidden = model.init_hidden(device)
            output, log_probs, y = model(ip)
        label_weights.append([y[0,0], y[0,1], y[0,2]])

        pred = torch.max(log_probs, 1)[1]
        if pred==gold:
            correct+=1
        else:
            incorrect+=1
        
        preds.append(pred)
        truth.append(gold)
            
    acc = correct/len(inf_data)
    return acc, np.array(label_weights, dtype=np.float)
    
def find_accuracy(pred, gold):
    acc=0
    for i in range(len(pred)):
        if pred[i]==gold[i]:
            acc+=1
    acc /= len(pred)
    return acc,y


def load_data(filename):
    data = pd.read_csv(filename).values
    data = data[:, 2:]
    dnaseq = data[:, 1]

    for d in range(len(dnaseq)):
        data[d, 1] = dnaseq[d][:400]
    return data


def prep_single_seq(seq, device):
    ip = []
    dnadict = {'A' :0, 'C': 1, 'G': 2, 'T': 3}
    for i in range(len(seq)):
        ip.append(dnadict[seq[i]])
    return torch.tensor(ip, dtype = torch.long, device=device)
            
def prep_single_label(label, device):

    gold = []
    gold.append(label)
    return torch.tensor(gold, dtype=torch.long, device=device)
