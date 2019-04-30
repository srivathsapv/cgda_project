import numpy as np
import pandas as pd
from tqdm import tqdm

import torch as t
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import Variable

import ml.utils as utils
from ml.model.vae.hybrid.batchloader import BatchLoader
from ml.model.vae.hybrid.parameters import Parameters
from ml.model.vae.hybrid.vae import VAE

RUN_OPTIONS = ["hybrid_vae_ordinal", "hybrid_vae_kmer_4", "hybrid_vae_kmer_5"]

def train_model(fpath_data, dirpath_results, use_gpu=True, verbose=True,
                args=None):

    feature_type = args.model_name.replace('hybrid_vae_', '')
    hyperparams = utils.get_model_hyperparams('hybrid_vae')
    logger = utils.get_logger(verbose)

    t.cuda.empty_cache()

    batch_loader = BatchLoader(data_path=fpath_data, is_kmer=('kmer' in feature_type))
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
        input, decoder_input, target = batch_loader.next_batch(hyperparams['batch_size'], 'train', use_gpu)
        target = target.view(-1)

        logits, aux_logits, kld = vae(hyperparams['dropout'], input, decoder_input)

        logits = logits.view(-1, batch_loader.vocab_size)
        cross_entropy = F.cross_entropy(logits, target, size_average=False)

        aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
        aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)

        loss = cross_entropy + hyperparams['aux'] * aux_cross_entropy + kld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        '''Validation'''
        input, decoder_input, target = batch_loader.next_batch(hyperparams['batch_size'], 'valid', use_gpu)
        target = target.view(-1)

        logits, aux_logits, valid_kld = vae(hyperparams['dropout'], input, decoder_input)

        logits = logits.view(-1, batch_loader.vocab_size)
        valid_cross_entropy = F.cross_entropy(logits, target, size_average=False)

        aux_logits = aux_logits.view(-1, batch_loader.vocab_size)
        valid_aux_cross_entropy = F.cross_entropy(aux_logits, target, size_average=False)

        loss = valid_cross_entropy + hyperparams['aux'] * valid_aux_cross_entropy + kld

        if iteration % 50 == 0:
            train_ce = cross_entropy.data.cpu().numpy()/(1024 * hyperparams['batch_size'])
            train_aux_ce = aux_cross_entropy.data.cpu().numpy()/(1024 * hyperparams['batch_size'])
            train_kl = kld.data.cpu().numpy()/(1024 * hyperparams['batch_size'])

            valid_ce = valid_cross_entropy.data.cpu().numpy()/(1024 * hyperparams['batch_size'])
            valid_aux_ce = valid_aux_cross_entropy.data.cpu().numpy()/(1024 * hyperparams['batch_size'])
            valid_kl = valid_kld.data.cpu().numpy()/(1024 * hyperparams['batch_size'])

            metrics.append({
                'train_ce': train_ce,
                'train_aux_ce': train_aux_ce,
                'train_kl': train_kl,
                'valid_ce': valid_ce,
                'valid_aux_ce': valid_aux_ce,
                'valid_kl': valid_kl
            })

            if valid_ce <= min_ce:
                min_vae_dict = vae.state_dict()
                min_ce = valid_ce
                logger.info('Saving best model in iteration {}'.format(iteration))
                t.save(vae.state_dict(), '{}/kmer-best.pth'.format(dirpath_results))

    fname = batch_loader.data_path.split('/')[-1].split('.')[0]
    logger.info('Saving final metrics')
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv('{}/metrics_{}.csv'.format(dirpath_results, fname))
