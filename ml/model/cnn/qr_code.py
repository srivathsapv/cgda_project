import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ml.model.cnn.data_loader import load_train_val_test_data
from ml.utils import get_logger

plt.style.use('seaborn')

warnings.filterwarnings("ignore")
torch.set_num_threads(1)

LOGGER = get_logger()

IMAGE_WIDTH = IMAGE_HEIGHT = 21  # 441 length zero-padded DNA sequences
IMAGE_CHANNELS = 4  # A, C, G, T

BASE_PAIR_MAP = {
    'A': [1, 0, 0, 0],
    'C': [0, 1, 0, 0],
    'G': [0, 0, 1, 0],
    'T': [0, 0, 0, 1],
    'X': [0, 0, 0, 0]
}

BASE_PAIR_COLORS = {
    (1, 0, 0, 0): [183, 28, 28],  # red
    (0, 1, 0, 0): [174, 234, 0],  # green
    (0, 0, 1, 0): [0, 145, 234],  # blue
    (0, 0, 0, 1): [255, 111, 0],  # orange
    (0, 0, 0, 0): [33, 33, 33]   # black
}

BASE_PAIR_CHAR = {
    (1, 0, 0, 0): "A",
    (0, 1, 0, 0): "C",
    (0, 0, 1, 0): "G",
    (0, 0, 0, 1): "T",
    (0, 0, 0, 0): "X"
}


def sequences_to_acgt_images(sequences):
    image = np.zeros((len(sequences), IMAGE_WIDTH,
                      IMAGE_HEIGHT, IMAGE_CHANNELS))
    for i, sequence in enumerate(sequences):
        for loc, base_pair in enumerate(sequence):
            row = loc // IMAGE_HEIGHT
            col = loc % IMAGE_HEIGHT
            image[i, row, col] = BASE_PAIR_MAP[base_pair]
    return image


def viz_dna_image(dna_image, save_path="", log=False):
    w, h, _ = dna_image.shape
    dna_rgb_image = np.zeros((w, h, 3))
    dna_char_image = np.empty((w, h), dtype="U10")
    for i, row in enumerate(dna_image):
        for j, col in enumerate(row):
            dna_rgb_image[i][j] = BASE_PAIR_COLORS[tuple(col)]
            dna_char_image[i][j] = BASE_PAIR_CHAR[tuple(col)]

    dna_rgb_image /= 255.0
    plt.grid(b=None)
    plt.imshow(dna_rgb_image)

    if log:
        LOGGER.info("DNA Sequence:\n" + str(dna_char_image.flatten()))
        LOGGER.info("DNA Block:\n" + str(dna_char_image))

    if save_path:
        plt.savefig(save_path)

    plt.clf()


def sequences_to_rgb_images(sequences, labels, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, "sample_rgb_images")):
        os.makedirs(os.path.join(save_path, "sample_rgb_images"))

    acgt_images = sequences_to_acgt_images(sequences)
    np.save(os.path.join(save_path, "acgt_images.npy"), acgt_images)
    np.save(os.path.join(save_path, "labels.npy"), labels)

    for i, dna_image in enumerate(acgt_images[:10]):
        viz_dna_image(dna_image, save_path=os.path.join(
            save_path, "sample_rgb_images", str(i) + ".jpg"), log=False)


def encode_and_dump(input_path, output_path):
    LOGGER.info("Extracting QRCode Features from DNA Sequences...")
    LOGGER.info("Note: When dumping the ACGT encoded 4-channel QRCode images as numpy arrays," +
                " we also save 10 samples of how these images would look as an RGB image in sample_rgb_images folder." +
                " The RGB encoding used: Red (A), Green (C), Blue (G), Orange (T), Black (N/A)")

    levels = ["phylum", "class", "order"]

    for level in tqdm(levels, total=len(levels)):
        data = load_train_val_test_data(input_path, level, analyze=False)
        def groupby(l, n): return [tuple(l[i:i+n])
                                   for i in range(0, len(l), n)]
        for split_name, (sequences, labels) in zip(["train", "val", "test"], groupby(data, 2)):
            sequences_to_rgb_images(sequences, labels, os.path.join(
                output_path, level, split_name))


if __name__ == '__main__':
    # # analyze sequences to get the image size
    # train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels = load_train_val_test_data("./data/hierarchy", "phylum", analyze=True)
    #
    # # Sample DNA Image Viz
    # sample_dna_sequence = "GACGATTAGTGGCXXX"  # 13bp length
    # dna_image = np.array([BASE_PAIR_MAP[c] for c in list(sample_dna_sequence)]).reshape(4, 4, 4)
    # viz_dna_image(dna_image, log=True)
    #
    # train_data = sequences_to_acgt_images(train_sequences)
    # viz_dna_image(train_data[0], log=True)
    #
    # sequences_to_rgb_images(train_sequences, save_path="./data/hierarchy")

    encode_and_dump("./data/hierarchy", "./data/cnn")
