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
                        help=('Name of the model to train. Possible values ' +
                              'are {}'.format(','.join(model_names))),
                        choices=set(model_names))

    parser.add_argument('--is-demo', action='store_true',
                        help='If True, train script is run in demo mode with limited number of epochs (typically 1 or 2)')

    return parser.parse_args()


def get_interface_module(model_name):
    for fpath_interface in glob('ml/**/interface.py', recursive=True):
        module_name = fpath_interface.replace('/', '.').replace('.py', '')
        interface_module = importlib.import_module(module_name)

        if model_name in interface_module.RUN_OPTIONS:
            return interface_module

    raise ValueError('Model not implemented')


def train():
    args = parse_args()

    if not args.model_name:
        raise ValueError('Model name is mandatory and should be one of - {}'.format(','.join(model_names)))

    data_dirpaths = ['cnn', 'hierarchy', 'vae', 'kmer']

    for data_dirpath in data_dirpaths:
        dirpath = os.path.join('data', data_dirpath)
        if not os.path.exists(dirpath):
            raise ValueError('Directory \'{}\' not found. Please run process_data.py before proceeding!'.format(dirpath))

    path_config = json.loads(open('config/paths.json', 'r').read())

    interface = get_interface_module(args.model_name)
    interface.train_model(path_config, args=args)


if __name__ == '__main__':
    train()
