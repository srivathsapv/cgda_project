import json
import logging

def get_model_hyperparams(model_name):
    return json.loads(open('config/hyperparams.json', 'r').read())[model_name]

def get_logger(verbose):
    level = logging.INFO if verbose else logging.DEBUG
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    return logging.getLogger(__name__)
