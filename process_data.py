from ml.preprocess.hierarchy import create_hierarchy
from ml.preprocess.kmer import generate_kmers
from ml.preprocess.vae import generate_vae_data

import ml.utils as utils

if __name__ == '__main__':
    fpath_taxa = 'data/taxa.csv'
    fpath_embeds = 'data/embeds.csv'

    fpath_hierarchy = 'data/hierarchy'
    fpath_kmer = 'data/kmer'
    dirpath_vae = 'data/vae'

    utils.init_logger(True)

    create_hierarchy(fpath_taxa, fpath_hierarchy)
    generate_kmers(fpath_hierarchy, fpath_kmer)
    generate_vae_data(fpath_embeds, dirpath_vae)
