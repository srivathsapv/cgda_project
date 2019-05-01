from ml.preprocess.hierarchy import create_hierarchy
from ml.preprocess.kmer import generate_kmers
from ml.preprocess.vae import generate_vae_data
from ml.model.cnn.qr_code import encode_and_dump

import ml.utils as utils

if __name__ == '__main__':
    fpath_taxa = 'data/taxa.csv'
    fpath_embeds = 'data/embeds.csv'

    fpath_hierarchy = 'data/hierarchy'
    fpath_kmer = 'data/kmer'
    dirpath_vae = 'data/vae'
    dirpath_cnn = 'data/cnn'

    utils.init_logger(True)

    create_hierarchy(fpath_taxa, fpath_hierarchy)
    generate_kmers(fpath_hierarchy, fpath_kmer)
    generate_vae_data(fpath_embeds, dirpath_vae)
    encode_and_dump(fpath_hierarchy, dirpath_cnn)
