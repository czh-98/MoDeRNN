import torch.nn as nn
import torch
from torch.autograd import Variable
from torch.nn import Module, Sequential, Conv2d


# -----------------------------------------  MoDeRNN Cell ---------------------------------------------------------------------------------
class MoDeRNN_cell(Module):

    def __init__(self, input_chans, hidden_size, filter_size, img_size, iterations=5):
        super(MoDeRNN_cell, self).__init__()

        # self.shape = shape #H, W
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.hidden_size = hidden_size

        self.padding = int((filter_size - 1) / 2)
        self._forget_bias = 1.0

        self.norm_cell = nn.LayerNorm([self.hidden_size, img_size, img_size])
        # Convolutional Layers
        self.conv_i2h = Sequential(
            Conv2d(self.input_chans, 4 * self.hidden_size, self.filter_size, 1, self.padding, bias=False),
            nn.LayerNorm([4 * self.hidden_size, img_size, img_size])
        )
        self.conv_h2h = Sequential(
            Conv2d(self.hidden_size, 4 * self.hidden_size, self.filter_size, 1, self.padding, bias=False),
            nn.LayerNorm([4 * self.hidden_size, img_size, img_size])
        )

        # MogConv
        self.iterations = iterations

        # hidden states buffer, [h, c]
        self.hiddens = None

        self.conv33_X = Sequential(
            Conv2d(self.input_chans, self.hidden_size, 3, 1, 1, bias=False),
            nn.LayerNorm([self.hidden_size, img_size, img_size])
        )

        self.conv55_X = Sequential(
            Conv2d(self.input_chans, self.hidden_size, 5, 1, 2, bias=False),
            nn.LayerNorm([self.hidden_size, img_size, img_size])
        )

        self.conv77_X = Sequential(
            Conv2d(self.input_chans, self.hidden_size, 7, 1, 3, bias=False),
            nn.LayerNorm([self.hidden_size, img_size, img_size])
        )

        self.conv33_H = Sequential(
            Conv2d(self.hidden_size, self.input_chans, 3, 1, 1, bias=False),
            nn.LayerNorm([self.input_chans, img_size, img_size])
        )

        self.conv55_H = Sequential(
            Conv2d(self.hidden_size, self.input_chans, 5, 1, 2, bias=False),
            nn.LayerNorm([self.input_chans, img_size, img_size])
        )

        self.conv77_H = Sequential(
            Conv2d(self.hidden_size, self.input_chans, 7, 1, 3, bias=False),
            nn.LayerNorm([self.input_chans, img_size, img_size])
        )

    def dcb(self, xt, ht):

        for i in range(1, self.iterations + 1):

            if i % 2 == 0:
                x33 = self.conv33_X(xt)
                x55 = self.conv55_X(xt)
                x77 = self.conv77_X(xt)

                x = (x33 + x55 + x77) / 3.0
                ht = 2 * torch.sigmoid(x) * ht

            else:
                h33 = self.conv33_H(ht)
                h55 = self.conv55_H(ht)
                h77 = self.conv77_H(ht)

                h = (h33 + h55 + h77) / 3.0
                xt = 2 * torch.sigmoid(h) * xt

        return xt, ht

    def forward(self, x, init_hidden=False):
        # initialize the hidden states, consists of hidden state: h and cell state: c
        if init_hidden or (self.hiddens is None):
            self.init_hiddens(x)

        h, c = self.hiddens

        x, h = self.dcb(x, h)

        # caculate i2h, h2h
        i2h = self.conv_i2h(x)
        h2h = self.conv_h2h(h)

        (i, f, g, o) = torch.split(i2h + h2h, self.hidden_size, dim=1)

        # caculate next h and c
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + self._forget_bias)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        self.gates_mean = [i.detach().mean(), f.detach().mean(), o.detach().mean()]
        self.gates_std = [i.detach().std(), f.detach().std(), o.detach().std()]

        next_c = f * c + i * g
        next_c = self.norm_cell(next_c)
        next_h = o * torch.tanh(next_c)

        self.hiddens = [next_h, next_c]
        return next_h

    def init_hiddens(self, x):
        b, c, h, w = x.size()

        self.hiddens = [Variable(torch.zeros(b, self.hidden_size, h, w)).cuda(),
                        Variable(torch.zeros(b, self.hidden_size, h, w)).cuda()]


class MoDeRNN(Module):
    def __init__(self, input_chans, output_chans, hidden_size=128, filter_size=5, num_layers=4, img_size=64):
        super(MoDeRNN, self).__init__()
        self.n_layers = num_layers
        # embedding layer
        self.embed = Conv2d(input_chans, hidden_size, 1, 1, 0)
        # lstm layers
        lstm = [MoDeRNN_cell(hidden_size, hidden_size, filter_size, img_size) for l in range(num_layers)]

        self.lstm = nn.ModuleList(lstm)
        # output layer
        self.output = Conv2d(hidden_size, output_chans, 1, 1, 0)
        self.state_hist_h = []
        self.state_hist_c = []

        self.att_hist_h = []
        self.att_hist_c = []

    def forward(self, x, init_hidden=False):
        h_in = self.embed(x)
        for l in range(self.n_layers):
            h_in = self.lstm[l](h_in, init_hidden)

        return self.output(h_in)


def get_MoDeRNN(input_chans=1, output_chans=1, hidden_size=64, filter_size=5, num_layers=4, img_size=64):
    model = MoDeRNN(input_chans, output_chans, hidden_size, filter_size, num_layers, img_size)
    return model
