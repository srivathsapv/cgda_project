import json
import logging

from ml.toggle_logger import ToggleLogger

def get_model_hyperparams(model_name):
    return json.loads(open('config/hyperparams.json', 'r').read())[model_name]

def get_logger(verbose):
    logger = ToggleLogger(date_format='%d-%b-%y %H:%M:%S', enabled=verbose)
    return logger
