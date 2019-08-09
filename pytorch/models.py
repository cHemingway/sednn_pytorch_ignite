''' Models for main_ignite.py. 
Every model takes n_concat, freq_bins as init arguments
New models are detected automatically '''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def move_data_to_gpu(x, cuda):

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. 
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing 
    human-level performance on imagenet classification." Proceedings of the 
    IEEE international conference on computer vision. 2015.
    """
    
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
        
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    
    
class DNN(nn.Module):
    def __init__(self, n_concat, freq_bins):
        
        super().__init__()
        
        hidden_units = 2048
        
        self.fc1 = nn.Linear(n_concat * freq_bins, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, freq_bins)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_layer(self.fc4)
        
    def forward(self, input):
        
        (batch_size, n_concat, freq_bins) = input.shape
        x = input.view(batch_size, n_concat * freq_bins)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.fc4(x)
        
        return x


class LSTM(nn.Module):
    ''' LSTM based on 'A new feature set for masking-based monaural speech seperation '''
    def __init__(self, n_concat, freq_bins):
        
        super().__init__()
        
        hidden_units = 1024
        self.lstm_size = 512
        self.lstm_layers = 2
        self.lstm_step = 32

        # From the paper, "The model consists of
        # Two LSTM layers of 512 units 
        self.lstm1 = nn.LSTM(freq_bins, self.lstm_size, num_layers=self.lstm_layers, batch_first=True)
        # Output layer, linear
        self.fc3 = nn.Linear(self.lstm_size, freq_bins)


    def init_hidden(self, x):
        ''' Reinitialise hidden state of LSTM. Call this once per batch '''
        h0 = torch.zeros(self.lstm_layers, x.size(0), self.lstm_size).to('cuda')
        c0 = torch.zeros(self.lstm_layers, x.size(0), self.lstm_size).to('cuda')
        return (h0, c0)
        
        
    def forward(self, input):
        (batch_size, n_concat, freq_bins) = input.shape

        # Select only middle bin, effectively ignoring n_concat
        # We do this by "cutting out" the n_concat axis
        midpoint = (n_concat // 2) + 1
        x = input.select(1,midpoint)

        # Format correctly for LSTM, format (batch,seq_len,input_size) 
        # We do this by using split, and then pad_sequence to clean up
        x = x.split(self.lstm_step)
        x = nn.utils.rnn.pad_sequence(x, batch_first=True)
        # x = x.reshape(-1, self.lstm_step, freq_bins)


        # Reset state of LSTM
        h0, c0 = self.init_hidden(x)

        # LSTM with 2 layers, activation/dropout internal
        # Therefore we pass in training state in case internal dropout is on
        x, _ = self.lstm1(x, (h0, c0))
        # Output layer, linear activation (i.e none)
        x = self.fc3(x)
        
        return x
