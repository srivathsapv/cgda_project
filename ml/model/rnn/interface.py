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

import ml.utils as utils

RUN_OPTIONS = ["rnn"]

def train_model(data_config, dirpath_results, use_gpu=False, verbose=True,args=None):

    if use_gpu:
        device=torch.device("cpu")
    else:
        device=torch.device("cpu")

    logger = utils.get_logger(verbose)

    embedding_dim = 50
    hidden_dim = 50
    learning_rate = 1e-3

    epochs = 50
    vocab_size = 4
    label_size = 3
    batch_size = 1
    train_data = load_data(data_config['phylumtrain'])
    valid_data = load_data(data_config['phylumval'])
    test_data = load_data(data_config['phylumtest'])
    model = LSTMClassifier(embedding_dim, hidden_dim, vocab_size, label_size, batch_size, device).to(device)
    loss_fun = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, epochs, device, logger, 'rnn_phylum')

    epochs=70
    vocab_size=4
    label_size=5
    batch_size=1
    train_data = load_data(data_config['classtrain'])
    valid_data = load_data(data_config['classval'])
    test_data = load_data(data_config['classtest'])
    model = LSTMClassifier(embedding_dim, hidden_dim, vocab_size, label_size, batch_size, device).to(device)
    loss_fun = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, epochs, device, logger, 'rnn_class')

    epochs=90
    vocab_size=4
    label_size=19
    batch_size=1
    train_data = load_data(data_config['ordertrain'])
    valid_data = load_data(data_config['orderval'])
    test_data = load_data(data_config['ordertest'])
    model = LSTMClassifier(embedding_dim, hidden_dim, vocab_size, label_size, batch_size, device).to(device)
    loss_fun = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, epochs, device, logger, 'rnn_order')
    


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, device):

        
        super(LSTMClassifier, self).__init__()
     
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.batch_size = batch_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.num_layers = 3
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional = True)
        
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        
        self.hidden = self.init_hidden(device)
        

    def init_hidden(self, device):
        return (autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim, device=device)),
                autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim, device=device)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size , -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return lstm_out[-1], log_probs


def train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, epochs, device, logger, exp_name):


    logdir = dirpath_results
    exp_dir = logdir + '/' + exp_name
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    start = time.time()
    metrics_file = exp_dir+'/metrics_best.tsv'
    with open(metrics_file,'w') as fout:
        for epoch in range(epochs):
            acc, loss = train_one_epoch(train_data[:2], model, loss_fun, optimizer, device)

            state = {'iter_num': epoch+1,
                     'enc_state': model.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     }
            filename = 'state_%010d.pt' % (epoch+1)
            save_file = exp_dir + '/' + filename
            torch.save(state, save_file)
            logger.info('wrote checkpoint to '+save_file)

            vacc = inference(valid_data, model, loss_fun, device)
            tacc = inference(test_data, model, loss_fun, device)
            print(acc, vacc, tacc, loss, sep='\t', file=fout)

    filename = 'bestmodel.pt'
    save_file = exp_dir + '/' + filename
    torch.save(state, save_file)
    logger.info('saved final model to '+save_file)


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











    


    
