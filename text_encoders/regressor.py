import torch
import torch.nn as nn
import torch.functional as F


class MLP(nn.Module):
    def __init__(self, in_size, h_size, out_size):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.h_size = h_size
        self.out_size = out_size

        self.fc1 = nn.Linear(in_size, h_size)
        self.fc2 = nn.Linear(h_size, out_size)

    def forward(self, texts_descriptors):
        h = F.relu(self.fc1(texts_descriptors))
        out = F.relu(self.fc2(h))
        return out


class RNN(nn.Module):
    def __init__(self, in_size, h_size, layer_name='gru', dropout=.2, num_layers=2,
                 bidirectional=True, device='gpu'):
        super(RNN, self).__init__()

        self.device = device
        self.layer_name = layer_name

        if layer_name == 'lstm':
            self.rnn = nn.LSTM(input_size=in_size, hidden_size=h_size, num_layers=num_layers, dropout=dropout,
                               batch_first=True, bidirectional=bidirectional)
        elif layer_name == 'gru':
            self.rnn = nn.GRU(input_size=in_size, hidden_size=h_size, num_layers=num_layers, dropout=dropout,
                              batch_first=True, bidirectional=bidirectional)
        else:
            self.rnn = nn.RNN(input_size=in_size, hidden_size=h_size, num_layers=num_layers, dropout=dropout,
                              batch_first=True, bidirectional=bidirectional, nonlinearity='tanh')

        self.h_size = h_size

    def __init_hidden(self, batch_size):
        return torch.zeros(4, batch_size, self.h_size, device=self.device)

    def forward(self, seqs_vectors):
        batch_size, seq_len, feats = seqs_vectors.size()
        if self.layer_name == 'lstm':
            h = self.__init_hidden(batch_size), self.__init_hidden(batch_size)
        else:
            h = self.__init_hidden(batch_size)

        output, h_n = self.rnn(seqs_vectors, h)

        if self.layer_name == 'lstm':
            return h_n[0]
        return h_n



