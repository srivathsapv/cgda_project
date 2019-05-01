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
from ml.model.rnn.model import LSTMClassifier
from ml.model.rnn.train import train
from ml.model.rnn.train import load_data
from ml.model.rnn.parameters import Parameters

RUN_OPTIONS = ["rnn"]

def train_model(path_config, args=None):

    if torch.cuda.is_available():
        device=torch.device("cuda")
    else:
        device=torch.device("cpu")
        
    logger = utils.get_logger()
    hyperparams = utils.get_model_hyperparams("rnn")
    learning_rate = hyperparams['learning_rate']
    path_config = path_config['rnn']
    dirpath_results = path_config['dirpath_rnn']
    

    logger.info('Training RNN Phylum Classifier')
    parameters = Parameters('phylum')
    data_config = path_config['phylum']
    train_data = load_data(data_config['train'])
    valid_data = load_data(data_config['val'])
    test_data = load_data(data_config['test'])
    model = LSTMClassifier(parameters.embedding_dim, parameters.hidden_dim, parameters.vocab_size, parameters.label_size, device).to(device)
    loss_fun = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, parameters, device, logger, args.is_demo, 'rnn_phylum')

    logger.info('Training RNN Class Classifier')
    parameters = Parameters('class')
    data_config = path_config['class']
    train_data = load_data(data_config['train'])
    valid_data = load_data(data_config['val'])
    test_data = load_data(data_config['test'])
    model = LSTMClassifier(parameters.embedding_dim, parameters.hidden_dim, parameters.vocab_size, parameters.label_size, device).to(device)
    loss_fun = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, parameters, device, logger, args.is_demo, 'rnn_class')

    logger.info('Training RNN Order Classifier')
    parameters = Parameters('order')
    data_config = path_config['order']
    train_data = load_data(data_config['train'])
    valid_data = load_data(data_config['val'])
    test_data = load_data(data_config['test'])
    model = LSTMClassifier(parameters.embedding_dim, parameters.hidden_dim, parameters.vocab_size, parameters.label_size, device).to(device)
    loss_fun = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = learning_rate)
    train(train_data, valid_data, test_data, model, loss_fun, optimizer, dirpath_results, parameters, device, logger, args.is_demo, 'rnn_order')
    









    


    
