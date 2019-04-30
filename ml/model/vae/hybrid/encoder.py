import torch as t
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embed_size, latent_size, kernel_params={}):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.latent_size = latent_size

        self.conv1 = nn.Conv1d(self.embed_size, 32, **kernel_params['conv1'])
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, **kernel_params['conv2'])
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 64, **kernel_params['conv3'])
        self.bn3 = self.bn2

        self.conv4 = nn.Conv1d(64, self.latent_size, **kernel_params['conv4'])
        self.bn4 = nn.BatchNorm1d(self.latent_size)

        self.elu = nn.ELU()

    def forward(self, input):
        """
        :param input: An float tensor with shape of [batch_size, seq_len, embed_size]
        :return: An float tensor with shape of [batch_size, latent_variable_size]
        """

        '''
        Transpose input to the shape of [batch_size, embed_size, seq_len]
        '''
        input = t.transpose(input, 1, 2)

        result = self.elu(self.conv1(input))
        result = self.elu(self.conv2(result))
        result = self.elu(self.conv3(result))

        result = self.elu(self.conv4(result))

        if result.shape[0] > 1 or result.shape[2] > 1:
            result = self.bn4(result)
        result = self.elu(result)

        return result.squeeze(2)
