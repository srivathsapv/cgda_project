import numpy as np
import pandas as pd
from tqdm import tqdm

import torch as t
from torch.optim import Adam
import torch.nn.functional as F

import ml.utils as utils
from ml.model.vae.hybrid.batchloader import BatchLoader
from ml.model.vae.hybrid.parameters import Parameters
from ml.model.vae.hybrid.vae import VAE

def get_metrics_dict(ce, aux_ce, kld, val_ce, val_aux_ce, val_kld, batch_size):
    train_ce = ce.data.cpu().numpy() / (1024 * batch_size)
    train_aux_ce = aux_ce.data.cpu().numpy() / (1024 * batch_size)
    train_kl = kld.data.cpu().numpy() / (1024 * batch_size)

    val_ce = val_ce.data.cpu().numpy() / (1024 * batch_size)
    val_aux_ce = val_aux_ce.data.cpu().numpy() / (1024 * batch_size)
    val_kl = val_kld.data.cpu().numpy() / (1024 * batch_size)

    return {
        'train_ce': train_ce,
        'train_aux_ce': train_aux_ce,
        'train_kl': train_kl,
        'valid_ce': val_ce,
        'valid_aux_ce': val_aux_ce,
        'valid_kl': val_kl
    }

def train_vae(path_config, feature_type, hyperparams, model_name):
    t.cuda.empty_cache()
    logger = utils.get_logger()

    is_kmer = ('kmer' in feature_type)
    fpath_data = (path_config['features']
                  if is_kmer else path_config['sequences'])
    dirpath_results = path_config['results']

    use_gpu = t.cuda.is_available()

    batch_loader = BatchLoader(data_path=fpath_data, is_kmer=(is_kmer))
    parameters = Parameters(batch_loader.vocab_size, feature_type=feature_type)

    vae = VAE(parameters)

    if use_gpu:
        vae = vae.cuda()

    optimizer = Adam(vae.parameters(), hyperparams['learning_rate'])

    metrics = []
    min_ce = 1000
    min_vae = None

    num = hyperparams['num_iterations']

    for iteration in tqdm(range(num), total=num):
        '''Train step'''
        input, decoder_input, target = batch_loader.next_batch(
            hyperparams['batch_size'], 'train', use_gpu)
        target = target.view(-1)

        logits, aux_logits, kld = vae(
            hyperparams['dropout'], input, decoder_input)

        logits = logits.view(-1, batch_loader.vocab_size)
        cross_entropy = F.cross_entropy(logits, target, size_average=False)

        aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
        aux_cross_entropy = F.cross_entropy(
            aux_logits, target, size_average=False)

        loss = cross_entropy + hyperparams['aux'] * aux_cross_entropy + kld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''Validation'''
        input, decoder_input, target = batch_loader.next_batch(
            hyperparams['batch_size'], 'valid', use_gpu)
        target = target.view(-1)

        logits, aux_logits, valid_kld = vae(
            hyperparams['dropout'], input, decoder_input)

        logits = logits.view(-1, batch_loader.vocab_size)
        valid_cross_entropy = F.cross_entropy(
            logits, target, size_average=False)

        aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
        valid_aux_cross_entropy = F.cross_entropy(
            aux_logits, target, size_average=False)

        loss = valid_cross_entropy + \
            hyperparams['aux'] * valid_aux_cross_entropy + kld

        if iteration % 50 == 0:
            metrics_dict = get_metrics_dict(
                cross_entropy, aux_cross_entropy, kld,
                valid_cross_entropy, valid_aux_cross_entropy, valid_kld,
                hyperparams['batch_size']
            )
            metrics.append(metrics_dict)

            valid_ce = metrics_dict['valid_ce']

            if valid_ce <= min_ce:
                min_vae_dict = vae.state_dict()
                min_ce = valid_ce
                logger.info(
                    'Saving best model in iteration {}'.format(iteration))
                t.save(vae.state_dict(),
                       '{}/{}_best.pth'.format(dirpath_results, model_name))

    logger.info('Saving final metrics')
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(
        '{}/{}_metrics.csv'.format(dirpath_results, model_name))
