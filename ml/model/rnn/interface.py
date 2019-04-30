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
from ml.model.rnn.model import LSTMClassifier
from ml.model.rnn.train import train
from ml.model.rnn.train import load_data
from ml.model.rnn.parameters import Parameters

RUN_OPTIONS = ["rnn"]

def train_model(data_config, dirpath_results, use_gpu=False, verbose=True,args=None):

    if use_gpu:
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
    logger = utils.get_logger(verbose)
    hyperparams = utils.get_model_hyperparams('rnn')
    learning_rate = hyperparams['learning_rate']

    parameters = Parameters('phylum')
    train_data = load_data(data_config['phylumtrain'])
    valid_data = load_data(data_config['phylumval'])
    test_data = load_data(data_config['phylumtest'])
    model = LSTMClassifier(parameters.embedding_dim, parameters.hidden_dim, parameters.vocab_size, parameters.label_size, parameters.batch_size, device).to(device)
    loss_fun = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, parameters.epochs, device, logger, 'rnn_phylum')

    parameters = Parameters('class')
    train_data = load_data(data_config['classtrain'])
    valid_data = load_data(data_config['classval'])
    test_data = load_data(data_config['classtest'])
    model = LSTMClassifier(parameters.embedding_dim, parameters.hidden_dim, parameters.vocab_size, parameters.label_size, parameters.batch_size, device).to(device)
    loss_fun = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, parameters.epochs, device, logger, 'rnn_class')

    parameters = Parameters('order')
    train_data = load_data(data_config['ordertrain'])
    valid_data = load_data(data_config['orderval'])
    test_data = load_data(data_config['ordertest'])
    model = LSTMClassifier(parameters.embedding_dim, parameters.hidden_dim, parameters.vocab_size, parameters.label_size, parameters.batch_size, device).to(device)
    loss_fun = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, parameters.epochs, device, logger, 'rnn_order')
    









    


    
