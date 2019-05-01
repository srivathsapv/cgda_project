from math import floor
import json


class Parameters:
    def __init__(self, vocab_size, feature_type):

        self.vocab_size = vocab_size
        self.embed_size = 16

        self.latent_size = 100

        self.decoder_rnn_size = 1000
        self.decoder_rnn_num_layers = 1

        fpath_params = 'config/hybrid_vae_kernel_params.json'
        self.kernel_params = json.loads(
            open(fpath_params, 'r').read())[feature_type]
