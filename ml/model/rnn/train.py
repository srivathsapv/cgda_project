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


def train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, parameters, device, logger, is_demo, exp_name):

    logdir = dirpath_results
    exp_dir = logdir + '/' + exp_name
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    metrics = []

    epochs = parameters.epochs
    batch_size = parameters.batch_size
    if is_demo:
        batch_size = 10
        valid_data = valid_data[:10]
        test_data = test_data[:10]
        epochs = 1

    for epoch in range(epochs):
        random.shuffle(train_data)
        acc, loss = train_one_epoch(
            train_data[:batch_size], model, loss_fun, optimizer, device)
        vacc = inference(valid_data, model, loss_fun, device)
        tacc = inference(test_data, model, loss_fun, device)
        metrics.append([acc, vacc, tacc, loss])
        logger.info('iter: {}, iter/n_iters: {}%'.format(epoch +
                                                         1, ((epoch+1) / epochs) * 100))

    if not is_demo:
        state = {'iter_num': epoch+1,
                 'enc_state': model.state_dict(),
                 'opt_state': optimizer.state_dict(),
                 }
        filename = 'bestmodel1.pt'
        save_file = exp_dir + '/' + filename
        metrics_file = exp_dir+'/metrics_best1.tsv'
        torch.save(state, save_file)
        write_metrics(metrics, metrics_file)
        logger.info('Saving final model to '+save_file)


def train_one_epoch(train_data, model, loss_fun, optimizer, device):

    optimizer.zero_grad()
    total_loss = 0
    correct = 0
    incorrect = 0
    truth = []
    preds = []
    for i, data in enumerate(train_data):
        seq = data[1]
        label = data[0]
        ip = prep_single_seq(seq, device)
        gold = prep_single_label(label, device)
        model.zero_grad()
        model.hidden = model.init_hidden(device)
        output, log_probs, _ = model(ip)
        loss = loss_fun(log_probs, gold)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        pred = torch.max(log_probs, 1)[1]
        if pred == gold:
            correct += 1
        else:
            incorrect += 1

        preds.append(pred)
        truth.append(gold)

    acc = correct/len(train_data)
    return acc, total_loss


def inference(inf_data, model, loss_fun, device):

    total_loss = 0
    correct = 0
    incorrect = 0
    truth = []
    preds = []
    for i, data in enumerate(inf_data):
        seq = data[1]
        label = data[0]
        ip = prep_single_seq(seq, device)
        gold = prep_single_label(label, device)
        with torch.no_grad():
            model.hidden = model.init_hidden(device)
            output, log_probs, _ = model(ip)
        loss = loss_fun(log_probs, gold)
        total_loss += loss

        pred = torch.max(log_probs, 1)[1]
        if pred == gold:
            correct += 1
        else:
            incorrect += 1

        preds.append(pred)
        truth.append(gold)

    acc = correct/len(inf_data)
    return acc


def find_accuracy(pred, gold):
    acc = 0
    for i in range(len(pred)):
        if pred[i] == gold[i]:
            acc += 1
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
    dnadict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    for i in range(len(seq)):
        ip.append(dnadict[seq[i]])
    return torch.tensor(ip, dtype=torch.long, device=device)


def prep_single_label(label, device):

    gold = []
    gold.append(label)
    return torch.tensor(gold, dtype=torch.long, device=device)


def write_metrics(metrics, metrics_file):

    with open(metrics_file, 'w') as fout:
        for m in metrics:
            print(m[0], m[1], m[2], m[3], sep='\t', file=fout)
