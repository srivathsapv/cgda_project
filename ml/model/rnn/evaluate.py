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
        load_state_file = sf

    if load_state_file is not None:
        state = torch.load(load_state_file, map_location=device)
        logger.info('Loaded checkpoint '+load_state_file)

    if state is not None:
        model.load_state_dict(state['enc_state'])

    acc,label_weights = inference(test_data, model, device)
    logger.info('Test accuracy: {}'.format(acc))
    
    '''
    print(np.shape(label_weights))
    test_labels=[]
    for data in test_data:
        test_labels.append(data[0])
    test_labels=np.array(test_labels)
    print(np.shape(test_labels))

    with open('inference2.tsv','w') as fout:
        for i in range(len(test_labels)):
            print(test_labels[i], label_weights[i, 0], label_weights[i, 1], label_weights[i, 2], sep='\t', file=fout)
    '''
    

    

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
        loss = loss_fun(log_probs, gold)
        total_loss += loss

        pred = torch.max(log_probs, 1)[1]
        if pred==gold:
            correct+=1
        else:
            incorrect+=1
        
        preds.append(pred)
        truth.append(gold)
            
    acc = correct/len(inf_data)
    return acc, np.array(label_weights)
    
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
