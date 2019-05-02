import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F


class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, device):

        super(LSTMClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.label_size = label_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)

        self.hidden2label = nn.Linear(hidden_dim*2, label_size)

        self.hidden = self.init_hidden(device)

    def init_hidden(self, device):
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim, device=device)),
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim, device=device)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return lstm_out[-1], log_probs, y
