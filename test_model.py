import os
import argparse
import importlib
from glob import glob
import json

import ml.utils as utils

model_names = ['basic_kmer', 'basic_vector', 'basic_onehot', 'cnn_qrcode', 'rnn', 'lstm_vae_ordinal', 'lstm_vae_kmer_4', 'lstm_vae_kmer_5',
               'hybrid_vae_ordinal', 'hybrid_vae_kmer_4',
               'hybrid_vae_kmer_5']


def parse_args():
    parser = argparse.ArgumentParser(description='Taxonomic Classification')

    parser.add_argument('--model-name', type=str,
                        help=('Name of the model to test. Possible values ' +
                              'are {}'.format(','.join(model_names))),
                        choices=set(model_names))

    return parser.parse_args()


def get_interface_module(model_name):
    for fpath_interface in glob('ml/**/interface.py', recursive=True):
        module_name = fpath_interface.replace('/', '.').replace('.py', '')
        interface_module = importlib.import_module(module_name)

        if model_name in interface_module.RUN_OPTIONS:
            return interface_module

    raise ValueError('Model not implemented')


def test():
    args = parse_args()

    if not args.model_name:
        raise ValueError(
            'Model name is mandatory and should be one of - {}'.format(','.join(model_names)))

    data_dirpaths = ['cnn', 'rnn', 'vae', 'basic']

    for data_dirpath in data_dirpaths:
        dirpath = os.path.join('results', data_dirpath)
        if not os.path.exists(dirpath):
            raise ValueError(
                'Directory \'{}\' not found. Please run train_model.py before proceeding!'.format(dirpath))

    path_config = json.loads(open('config/paths.json', 'r').read())

    interface = get_interface_module(args.model_name)
    if not hasattr(interface, 'test_model'):
        raise AttributeError('Model testing for {} not implemented'.format(args.model_name))

    interface.test_model(path_config, args=args)

if __name__ == '__main__':
    test()
