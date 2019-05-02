import ml.utils as utils
from ml.model.vae.hybrid.train_helper import train_vae
from ml.model.vae.hybrid.test_helper import test_vae

RUN_OPTIONS = ["hybrid_vae_ordinal", "hybrid_vae_kmer_4", "hybrid_vae_kmer_5"]


def get_helper_params(path_config, args=None):
    feature_type = args.model_name.replace('hybrid_vae_', '')
    hyperparams = utils.get_model_hyperparams('hybrid_vae')

    path_config = path_config[args.model_name]
    return path_config, feature_type, hyperparams


def train_model(path_config, args=None):
    path_config, feature_type, hyperparams = get_helper_params(
        path_config, args)

    if args.is_demo:
        hyperparams['num_iterations'] = 1
    train_vae(path_config, feature_type, hyperparams, args.model_name)


def test_model(path_config, args=None):
    path_config, feature_type, hyperparams = get_helper_params(
        path_config, args)
    test_vae(path_config, feature_type, hyperparams, args.model_name)
