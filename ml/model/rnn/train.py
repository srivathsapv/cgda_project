import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import logging
import pandas as pd
import glob
import os
import numpy as np

import ml.utils as utils

def train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, epochs, device, logger, exp_name):


    logdir = dirpath_results
    exp_dir = logdir + '/' + exp_name
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    metrics_file = exp_dir+'/metrics_best.tsv'
    with open(metrics_file,'w') as fout:
        for epoch in range(epochs):
            random.shuffle(train_data)
            acc, loss = train_one_epoch(train_data[:1000], model, loss_fun, optimizer, device)
            vacc = inference(valid_data, model, loss_fun, device)
            tacc = inference(test_data, model, loss_fun, device)
            print(acc, vacc, tacc, loss, sep='\t', file=fout)
            logger.info('iter: {}, iter/n_iters: {}%'.format(epoch+1, ((epoch+1) / epochs) * 100))

    state = {'iter_num': epoch+1,
             'enc_state': model.state_dict(),
             'opt_state': optimizer.state_dict(),
                     }
    filename = 'bestmodel.pt'
    save_file = exp_dir + '/' + filename
    torch.save(state, save_file)
    logger.info('Saving final model to '+save_file)


def train_one_epoch(train_data, model, loss_fun, optimizer, device):

    optimizer.zero_grad()
    total_loss=0
    correct=0
    incorrect=0
    truth=[]
    preds=[]
    for i, data in enumerate(train_data):
        seq = data[1]
        label = data[0]
        ip = prep_single_seq(seq, device)
        gold = prep_single_label(label, device)
        model.zero_grad()
        model.hidden = model.init_hidden(device)
        output, log_probs = model(ip)          
        loss = loss_fun(log_probs, gold)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = torch.max(log_probs, 1)[1]
        if pred==gold:
            correct+=1
        else:
            incorrect+=1

        preds.append(pred)
        truth.append(gold)
            
    acc = correct/len(train_data)
    return acc, total_loss
    

def inference(inf_data, model, loss_fun, device):

    total_loss=0
    correct=0
    incorrect=0
    truth=[]
    preds=[]
    for i, data in enumerate(inf_data):
        seq = data[1]
        label = data[0]
        ip = prep_single_seq(seq, device)
        gold = prep_single_label(label, device)
        with torch.no_grad():
            model.hidden = model.init_hidden(device)
            output, log_probs = model(ip)          
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
    return acc
    
def find_accuracy(pred, gold):
    acc=0
    for i in range(len(pred)):
        if pred[i]==gold[i]:
            acc+=1
    acc /= len(pred)
    return acc


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

