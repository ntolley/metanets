import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
import os
from sklearn.preprocessing import StandardScaler

# LSTM autoencoder modules
class model_lstm(nn.Module):
    def __init__(self, input_size, hidden_dim, n_layers, dropout, device='cpu', bidirectional=False):
        super(model_lstm, self).__init__()

        #multiplier based on bidirectional parameter
        if bidirectional:
            num_directions = 2
        else:
            num_directions = 1

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers * num_directions
        self.device = device
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.input_size = input_size

        self.lstm = nn.LSTM(self.input_size, self.hidden_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional) 
    
    def forward(self, x):
        batch_size = x.size(0)
        # Initializing hidden state for first input using method defined below
        hidden, cell = self.init_hidden(batch_size)

        out, (hidden, cell) = self.lstm(x, (hidden, cell))

        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data.to(self.device)

        #LSTM initialization
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device)
        cell = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(self.device) + 1

        return hidden, cell


class model_lstm_autoencoder(nn.Module):
    def __init__(self, input_size, lstm_encoder_hidden, lstm_decoder_hidden,
                 encoder_layer_size, decoder_layer_size, bottleneck=10, device='cpu'):
        super(model_lstm_autoencoder, self).__init__()
        self.input_size = input_size
        self.encoder_layer_size, self.decoder_layer_size = encoder_layer_size, decoder_layer_size
        self.lstm_encoder_hidden, self.lstm_decoder_hidden = lstm_encoder_hidden, lstm_decoder_hidden
        self.bottleneck = bottleneck

        self.encoder_lstm = model_lstm(input_size=input_size, hidden_dim=lstm_encoder_hidden, n_layers=2, dropout=0.1)
        self.encoder_ann = model_ann(input_size, bottleneck, layer_size=self.encoder_layer_size)
        self.decoder_ann = model_ann(bottleneck, input_size, layer_size=self.decoder_layer_size)
        self.decoder_lstm = model_lstm(input_size=input_size, hidden_dim=lstm_encoder_hidden, n_layers=2, dropout=0.1)

        self.fc = nn.Linear(self.input_size, self.input_size)

    def forward(self, x):
        out = self.encoder_lstm(x)
        out = out.contiguous()
        out = self.encoder_ann(out)

        out = self.decoder_ann(out)
        out = self.decoder_lstm(out)
        out = self.fc(out)
        return out


# Feed forward network autoencoder modules
class model_ann(nn.Module):
    def __init__(self, input_size, output_size, layer_size, dropout=0.0):
        super(model_ann, self).__init__()
        self.input_size,  self.layer_size, self.output_size = input_size, layer_size, output_size

        #List layer sizes
        self.layer_hidden = np.concatenate([[input_size], layer_size, [output_size]])
        
        #Compile layers into lists
        self.layer_list = nn.ModuleList(
            [nn.Linear(in_features=self.layer_hidden[idx], out_features=self.layer_hidden[idx+1]) for idx in range(len(self.layer_hidden)-1)] )

        self.dropout_list = [nn.Dropout(p=dropout) for _ in range(len(self.layer_list)-1)]
 
    def forward(self, x):
        #Encoding step
        for idx in range(len(self.layer_list) - 1):
            x = torch.tanh(self.layer_list[idx](x))
            x = self.dropout_list[idx](x)
        x = self.layer_list[-1](x)
        return x

class model_ann_autoencoder(nn.Module):
    def __init__(self, input_size, output_size, encoder_layer_size, decoder_layer_size, bottleneck = 10):
        super(model_ann_autoencoder, self).__init__()
        self.input_size, self.output_size = input_size, output_size
        self.encoder_layer_size, self.decoder_layer_size = encoder_layer_size, decoder_layer_size

        self.encoder = model_ann(input_size, bottleneck, layer_size=self.encoder_layer_size)
        self.decoder = model_ann(bottleneck, input_size, layer_size=self.decoder_layer_size)

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)

        return out

#Helper function to pytorch train networks for decoding
def train_validate_model(model, optimizer, criterion, max_epochs, training_generator, validation_generator, device, print_freq=10, early_stop=20):
    train_loss_array = []
    validation_loss_array = []
    # Loop over epochs
    min_validation_loss, min_validation_std, min_validation_counter, min_validation_epoch = np.inf, np.inf, 0, 0
    for epoch in range(max_epochs):
        #___Train model___
        model.train()
        train_batch_loss = []
        validation_batch_loss = []
        for batch_x in training_generator:
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            batch_x = batch_x.float().to(device)

            output = model(batch_x)
            train_loss = criterion(output, batch_x)
            train_loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            train_batch_loss.append(train_loss.item())
        
        train_loss_array.append(train_batch_loss)

        #___Evaluate Model___
        with torch.no_grad():
            model.eval()
            #Generate train set predictions
            for batch_x in validation_generator:
                batch_x = batch_x.float().to(device)

                output = model(batch_x)
                validation_loss = criterion(output, batch_x)

                validation_batch_loss.append(validation_loss.item())

        validation_loss_array.append(validation_batch_loss)

        #Compute average loss on batch
        train_epoch_loss = np.mean(train_batch_loss)
        train_epoch_std = np.std(train_batch_loss)
        validation_epoch_loss = np.mean(validation_batch_loss)
        validation_epoch_std = np.std(validation_batch_loss)

       #Check if validation loss reaches minimum 
        if validation_epoch_loss < min_validation_loss:
            print('*',end='')
            min_validation_loss = np.copy(validation_epoch_loss)
            min_validation_std = np.copy(validation_epoch_std)
            min_validation_counter = 0
            min_validation_epoch = np.copy(epoch+1)

            min_train_loss = np.copy(train_epoch_loss)
            min_train_std = np.copy(train_epoch_std)
            
        else:
            print('.',end='')
            min_validation_counter += 1

        #Print Loss Scores
        if (epoch+1)%print_freq == 0:
            print('')
            print('Epoch: {}/{} ...'.format(epoch+1, max_epochs), end=' ')
            print('Train Loss: {:.4f}  ... Validation Loss: {:.4f}'.format(train_epoch_loss,validation_epoch_loss))
        
        #Early stop if no validation improvement over set number of epochs
        if min_validation_counter > early_stop:
            print(' Early Stop; Min Epoch: {}'.format(min_validation_epoch))
            break

    loss_dict = {'min_validation_loss':min_validation_loss, 'min_validation_std':min_validation_std,'min_validation_epoch':min_validation_epoch, 
    'min_train_loss':min_train_loss, 'min_train_std':min_train_std,
    'train_loss_array':train_loss_array, 'validation_loss_array':validation_loss_array, 'max_epochs':max_epochs}
    return loss_dict