import argparse
import importlib
from glob import glob
import json

import ml.utils as utils

def parse_args():
    parser = argparse.ArgumentParser(description='Taxonomic Classification')

    model_names = ['basic_kmer', 'basic_vector', 'basic_onehot', 'cnn_qrcode', 'rnn', 'lstm_vae',
                   'hybrid_vae_ordinal', 'hybrid_vae_kmer_4',
                   'hybrid_vae_kmer_5']

    parser.add_argument('--model-name', type=str,
                        help=('Name of the model to train. Possible values ' +
                              'are {}'.format(','.join(model_names))),
                        choices=set(model_names))

    parser.add_argument('--output-path', type=str, help='Path to store training artifacts - network weights, analysis plots etc')

    parser.add_argument('--use-gpu', type=bool, default=False,
                        help='If True, GPU device is used for training (default: False)')

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

    data_config = json.loads(open('config/data_files.json', 'r').read())

    interface = get_interface_module(args.model_name)
    interface.train_model(data_config, args.output_path,
                          args.use_gpu, verbose=True, args=args)

if __name__ == '__main__':
    train()
