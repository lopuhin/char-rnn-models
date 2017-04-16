import torch
import torch.nn as nn

from utils import variable


class CharRNN(nn.Module):
    cell_cls = None

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(CharRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.cell = self.cell_cls(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        batch_size = inputs.size(0)
        encoded = self.encoder(inputs)
        output, hidden = self.cell(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden


class CharGRU(CharRNN):
    cell_cls = nn.GRU

    def init_hidden(self, batch_size):
        return variable(
            torch.zeros(self.n_layers, batch_size, self.hidden_size))


class CharLSTM(CharRNN):
    cell_cls = nn.LSTM

    def init_hidden(self, batch_size):
        return (
            variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
            variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
