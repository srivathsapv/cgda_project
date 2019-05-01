from math import floor
import json


class Parameters:
    def __init__(self, vocab_size, feature_type):
        fpath_params = 'config/hybrid_vae_kernel_params.json'
        self.kernel_params = json.loads(
            open(fpath_params, 'r').read())[feature_type]

        fpath_hyperparams = 'config/hyperparams.json'
        self.hyperparams = json.loads(open(fpath_hyperparams, 'r').read())['hybrid_vae']

        self.vocab_size = vocab_size
        self.embed_size = self.hyperparams['embed_size']

        self.latent_size = self.hyperparams['latent_size']

        self.decoder_rnn_size = self.hyperparams['decoder_rnn_size']
        self.decoder_rnn_num_layers = self.hyperparams['decoder_rnn_num_layers']
