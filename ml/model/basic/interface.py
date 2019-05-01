from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from ml.model.basic.kmer import train_basic as kmer_train_basic
from ml.model.basic.vectorized import train_basic as vector_train_basic
from ml.model.basic.onehot import train_basic as onehot_train_basic
import ml.utils as utils

RUN_OPTIONS = ['basic_kmer', 'basic_vector', 'basic_onehot']

def train_model(path_config, args=None):
   if args.model_name == 'basic_kmer':
       path_config = path_config['basic_kmer']
       kmer_config = utils.get_model_hyperparams('basic_kmer')

       models = [
           SVC(**kmer_config['svm']),
           RandomForestClassifier(**kmer_config['random_forest'])
       ]

       kmer_train_basic(
           models, dirpath_kmer=path_config['dirpath_kmer'],
           dirpath_output=path_config['results'], kmin=kmer_config['kmin'],
           kmax=kmer_config['kmax']
       )
   elif args.model_name == 'basic_vector':
       path_config = path_config['basic_vector']
       vector_train_basic(
           dirpath_vector=path_config['dirpath_vector'],
           dirpath_output=path_config['results']
       )

   elif args.model_name == 'basic_onehot':
       path_config = path_config['basic_onehot']
       onehot_train_basic(
           dirpath_vector=path_config['dirpath_onehot'],
           dirpath_output=path_config['results']
       )
   else:
       raise ValueError('Basic ML model {} not supported'.format(args.model_name))
