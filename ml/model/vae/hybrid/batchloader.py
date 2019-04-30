import numpy as np
import torch as t
from torch.autograd import Variable

class BatchLoader:
    def __init__(self, data_path, is_kmer=False):
        """
        :param data_path: string prefix to path of data folder
        :param force_preprocessing: whether to force data preprocessing
        """

        assert isinstance(data_path, str), \
            'Invalid data_path_prefix type. Required {}, but {} found'.format(str, type(data_path))

        self.split = 6500

        self.data_path = data_path
        self.is_kmer = is_kmer

        '''
        go_token (stop_token) uses to mark start (end) of the sequence while decoding
        pad_token uses to fill tensor to fixed-size length
        '''
        self.go_token = '>'
        self.pad_token = ''
        self.stop_token = '<'

        """
        performs data preprocessing
        """

        if self.is_kmer:
            data = np.load(self.data_path).astype(np.float16)
            self.vocab_size = data.shape[1]
        else:
            data = open(self.data_path, 'r', encoding='utf-8').read()

            self.vocab_size, self.idx_to_char, self.char_to_idx = self.build_vocab(data)

            self.max_seq_len = 440
            data = np.array([[self.char_to_idx[char] for char in line] for line in data.split('\n')[:-1]
                             if 400 <= len(line) <= self.max_seq_len])

        self.valid_data, self.train_data = data[:self.split], data[self.split:]
        self.data_len = [len(var) for var in [self.train_data, self.valid_data]]

    def build_vocab(self, data):

        # unique characters with blind symbol
        chars = sorted(list(set(data)) + [self.pad_token, self.go_token, self.stop_token])

        chars_vocab_size = len(chars)

        # mappings itself
        idx_to_char = chars
        char_to_idx = {x: i for i, x in enumerate(idx_to_char)}

        return chars_vocab_size, idx_to_char, char_to_idx

    def next_batch(self, batch_size, target: str, use_cuda=False):
        """
        :param batch_size: num_batches to lockup from data
        :param target: if target == 'train' then train data uses as target, in other case test data is used
        :param use_cuda: whether to use cuda
        :return: encoder and decoder input
        """

        """
        Randomly takes batch_size of lines from target data
        and wrap them into ready to feed in the model Tensors
        """

        target = 0 if target == 'train' else 1

        indexes = np.array(np.random.choice(self.data_len[target], size=batch_size, replace=False))

        encoder_input = [np.copy([self.train_data, self.valid_data][target][idx]).tolist() for idx in indexes]

        return self._wrap_tensor(encoder_input, use_cuda)

    def _wrap_tensor(self, input, use_cuda: bool):
        """
        :param input: An list of batch size len filled with lists of input indexes
        :param use_cuda: whether to use cuda
        :return: encoder_input, decoder_input and decoder_target tensors of Long type
        """

        """
        Creates decoder input and target from encoder input
        and fills it with pad tokens in order to initialize Tensors
        """
        if self.is_kmer:
            result = [np.array(var) for var in [input, input, input]]
            result = [Variable(t.from_numpy(var)).long() for var in result]
        else:
            batch_size = len(input)

            '''Add go token before decoder input and stop token after decoder target'''
            encoder_input = [[self.char_to_idx[self.go_token]] + list(line) for line in np.copy(input)]
            decoder_input = [[self.char_to_idx[self.go_token]] + list(line) for line in np.copy(input)]
            decoder_target = [line + [self.char_to_idx[self.stop_token]] for line in np.copy(input)]


            '''Evaluate how much it is necessary to fill with pad tokens to make the same lengths'''
            to_add = [self.max_seq_len - len(input[i]) for i in range(batch_size)]

            for i in range(batch_size):
                to_add_arr = [self.char_to_idx[self.pad_token]] * to_add[i]
                encoder_input[i] = list(encoder_input[i])
                decoder_input[i] = list(decoder_input[i])
                decoder_target[i] = list(decoder_target[i])

                encoder_input[i].extend(to_add_arr)
                decoder_input[i].extend(to_add_arr)
                decoder_target[i].extend(to_add_arr)

            result = [np.array(var) for var in [encoder_input, decoder_input, decoder_target]]
            result = [Variable(t.from_numpy(var)).long() for var in result]

        if use_cuda:
            result = [var.cuda() for var in result]

        return tuple(result)
