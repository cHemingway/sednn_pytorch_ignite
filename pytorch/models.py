''' Models for main_ignite.py. 
Every model takes n_concat, freq_bins as init arguments
New models are detected automatically

Add hyperparameters (numerical only) as keyword only arguments, and they will
be added to the command line parser, e.g
    class MyModel(nn.Module)
        __init__(self, n_concat, freq_bins, * , num_layers=4):
            pass
    
    Ends up with this call signature on the command line:
        main_ignite.py MyModel train --num_layers=4

 '''

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

    
class DNN(nn.Module):
    def __init__(self, n_concat, freq_bins, *, dropout=0.2):
        
        super().__init__()
        
        hidden_units = 2048
        self.dropout = dropout
        
        self.fc1 = nn.Linear(n_concat * freq_bins, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, freq_bins)
        
        self.init_weights()

    @staticmethod
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
    
    @staticmethod
    def init_bn(bn):
        """Initialize a Batchnorm layer. """
        
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.)
        
    def init_weights(self):
        
        self.init_layer(self.fc1)
        self.init_layer(self.fc2)
        self.init_layer(self.fc3)
        self.init_layer(self.fc4)
        
    def forward(self, input):
        
        (batch_size, n_concat, freq_bins) = input.shape
        x = input.view(batch_size, n_concat * freq_bins)
        
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc4(x)
        
        return x


class LSTM(nn.Module):
    ''' LSTM based on 'A new feature set for masking-based monaural speech seperation '''
    def __init__(self, n_concat, freq_bins, *, dropout=0.2, hidden_units = 1024,
                timestep=64, rnn_size = 512):
        
        super().__init__()
        
        self.dropout = dropout
        self.lstm_size = rnn_size
        self.lstm_timestep = timestep # Timestep of LSTM, and size of mini_batch

        # From the paper, "The model consists of
        # One fully connected layer of size 1024
        self.fc1 = nn.Linear(freq_bins, hidden_units)
        # One hidden layer
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        # Two LSTM layers of 512 units 
        self.lstm1 = nn.LSTMCell(hidden_units, self.lstm_size)
        self.lstm2 = nn.LSTMCell(self.lstm_size, self.lstm_size)
        # Output layer, linear
        self.fc3 = nn.Linear(self.lstm_size, freq_bins)


    def init_hidden(self, input_size):
        ''' Reinitialise hidden state of LSTM. Call this once per batch seq '''
        h0 = torch.zeros(input_size, self.lstm_size).to('cuda')
        c0 = torch.zeros(input_size, self.lstm_size).to('cuda')
        return (h0, c0)
        
        
    def forward(self, input):
        (batch_size, n_concat, freq_bins) = input.shape

        # Select only middle bin, effectively ignoring n_concat
        # We do this by "cutting out" the n_concat axis
        midpoint = (n_concat // 2) + 1
        batch = input.select(1,midpoint)

        # Reset state of LSTM per minibatch
        h0, c0 = self.init_hidden(self.lstm_timestep)
        h1, c1 = self.init_hidden(self.lstm_timestep)

        # Split input into minibatches
        # Note: The last minibatch may be shorter if batch is not evenly divisable
        minibatches = batch.split(self.lstm_timestep)

        output = []

        for x in minibatches:
            if x.size(0) != self.lstm_timestep:
                # HACK: Zero-Pad the shorter final minibatch
                pad_size = self.lstm_timestep - x.size(0)
                x = torch.nn.functional.pad(x, (0,0,0,pad_size))

            # Input layer
            x = self.fc1(x)
            # Fully connected hidden layer
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            # 2 LSTM Layers
            h0, c0 = self.lstm1(x, (h0, c0))
            x = F.dropout(h0, p=self.dropout, training=self.training)
            h1, c1 = self.lstm2(x, (h1, c1))
            x = F.dropout(h1, p=self.dropout, training=self.training)
            # Output layer, linear activation (i.e none)
            x = self.fc3(x)

            output.append(x)
            
        # Concat output together
        output = torch.cat(output)
        # Remove the padding from the end
        output = output[:batch_size,:]
        return output # Return concatanated version


class GRU(nn.Module):
    ''' GRU model, supports variable length '''
    def __init__(self, n_concat, freq_bins, 
                  *, dropout=0.2, hidden_units=1024, timestep=64, rnn_layers=2, rnn_size=512):
    
        super().__init__()
        
        self.dropout = dropout
        self.timestep = timestep
        self.rnn_size = rnn_size
        self.rnn_layers = rnn_layers
        self.hidden_units = hidden_units

        # From the paper, "The model consists of
        # One fully connected layer of size 1024
        self.fc1 = nn.Linear(freq_bins, self.hidden_units)
        # One hidden layer
        self.fc2 = nn.Linear(self.hidden_units, self.hidden_units)
        # N GRU Layers
        self.gru1 = nn.GRU(input_size=self.hidden_units, 
                           hidden_size=self.rnn_size, 
                           num_layers=self.rnn_layers)
        # Output layer
        self.fc3 = nn.Linear(self.rnn_size, freq_bins)

                
    def forward(self, input):
        (batch_size, n_concat, freq_bins) = input.shape

        # Select only middle bin, effectively ignoring n_concat
        # We do this by "cutting out" the n_concat axis
        midpoint = (n_concat // 2) + 1
        x = input.select(1,midpoint)

        # Run fully connected nets
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Reshape into sequence, batch, freq_bins format
        x = torch.split(x, self.timestep)
        x = nn.utils.rnn.pad_sequence(x)

        gru_batch_size = x.shape[1] # Need batch size for hidden state

        # Run GRU with initial random state
        # Don't keep state around for next timestep
        h0 = torch.randn(self.rnn_layers, gru_batch_size, self.rnn_size, 
                         device=x.device)
        x, h0 = self.gru1(x, h0)
        
        # Unpack and concatanate back, trimming off padding
        x = torch.reshape(x, (-1,self.rnn_size))
        x = x[:batch_size,:]
        
        # Run fully connected net for output
        x = self.fc3(x)

        return x
