import sys
sys.path.append('../code/')
import utils
from functools import partial

import numpy as np
import pandas as pd
import pickle

import torch
from torch import nn
from sklearn.model_selection import ShuffleSplit

device = torch.device("cuda:0")
# device = torch.device("cpu")

def get_generators(df, cv_dict, fold, batch_size=10_000, num_cores=1):

    train_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_cores, 'pin_memory':False}
    train_eval_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_cores, 'pin_memory':False}
    validation_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': num_cores, 'pin_memory':False}
    test_params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': num_cores, 'pin_memory':False}

    # Generators
    training_set = utils.fmri_dataset(cv_dict, fold, 'train_idx', df, scaler=None)
    scaler = training_set.scaler

    training_generator = torch.utils.data.DataLoader(training_set, **train_params)
    training_eval_generator = torch.utils.data.DataLoader(training_set, **train_eval_params)

    validation_set = utils.fmri_dataset(cv_dict, fold, 'validation_idx', df, scaler=scaler)
    validation_generator = torch.utils.data.DataLoader(validation_set, **validation_params)

    testing_set = utils.fmri_dataset(cv_dict, fold, 'test_idx', df, scaler=scaler)
    testing_generator = torch.utils.data.DataLoader(testing_set, **test_params)

    data_arrays = (training_set, validation_set, testing_set)
    generators = (training_generator, training_eval_generator, validation_generator, testing_generator)

    return generators, data_arrays

def run_mamba(df, cv_dict, fold, bottleneck=15, lr=1e-3, hidden_size=100, weight_decay=0.0):
    # Get training generators
    generators, data_arrays = get_generators(df, cv_dict, fold)

    training_generator, _, validation_generator, _ = generators
    training_set, _, _ = data_arrays

    #Define hyperparameters
    max_epochs = 1000
    input_size = training_set[0].shape[1]
    criterion = nn.MSELoss()
    # criterion = partial(utils.cov_mse_loss, cov_loss_weight=cov_loss_weight)

    encoder_hidden = hidden_size
    decoder_hidden = encoder_hidden

    model = utils.model_mamba_autoencoder(input_size, encoder_hidden, decoder_hidden,
                                    bottleneck=bottleneck, device=device).to(device)

    # Define Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #Train model
    loss_dict = utils.train_validate_model(model, optimizer, criterion, max_epochs, training_generator, validation_generator, device, 10, 30)

    # Save metadata
    metadata = {'cv_dict': cv_dict, 'lr': lr, 'weight_decay': weight_decay, 'max_epochs': max_epochs,
                'input_size': input_size, 'bottleneck': bottleneck, 'encoder_hidden': encoder_hidden, 'decoder_hidden': decoder_hidden}

    loss_dict['metadata'] = metadata

    return loss_dict, model

def bottleneck_sweep(df, cv_dict, num_folds):
    # Sweep through bottleneck values
    bottleneck_values = [1,3,5,7,9,11,13,15]
    # bottleneck_values = [15]

    train_results = dict()
    train_results['bottleneck_values'] = bottleneck_values

    for bottleneck in bottleneck_values:
        train_results[f'bottleneck_{bottleneck}'] = dict()
        for fold in range(num_folds):
            print(f'Training model on fold {fold}; bottleneck: {bottleneck}', end='\n')

            # Run one training instance
            res_dict, model = run_mamba(df=df, cv_dict=cv_dict, fold=fold, bottleneck=bottleneck)
            print(' ')

            # Save model
            torch.save(model.state_dict(), f'../models/mamba/bottleneck/mamba_fold{fold}_bottleneck{bottleneck}.pt')

            # Save results on every loop in case early stop
            train_results[f'bottleneck_{bottleneck}'][f'fold_{fold}'] = res_dict
            #Save metadata
            output = open(f'../data/mamba_bottleneck_sweep_results.pkl', 'wb')
            pickle.dump(train_results, output)
            output.close()

def learning_rate_sweep(df, cv_dict, num_folds):
    # Sweep through learning rate values
    lr_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

    train_results = dict()
    train_results['learning_rate_values'] = lr_values

    for lr in lr_values:
        train_results[f'lr_{lr}'] = dict()
        for fold in range(num_folds):
            print(f'Training model on fold {fold}; lr: {lr}', end='\n')

            # Run one training instance
            res_dict, model = run_mamba(df=df, cv_dict=cv_dict, fold=fold, lr=lr)
            print(' ')

            # Save model
            torch.save(model.state_dict(), f'../models/mamba/learning_rate/mamba_fold{fold}_lr{lr}.pt')

            # Save results on every loop in case early stop
            train_results[f'lr_{lr}'][f'fold_{fold}'] = res_dict
            #Save metadata
            output = open(f'../data/mamba_lr_sweep_results.pkl', 'wb')
            pickle.dump(train_results, output)
            output.close()

def weight_decay_sweep(df, cv_dict, num_folds):
    # Sweep through learning rate values
    weight_decay_values = [1e-2, 1e-3, 1e-4, 1e-5, 0.0]

    train_results = dict()
    train_results['weight_decay_values'] = weight_decay_values

    for weight_decay in weight_decay_values:
        train_results[f'weight_decay_{weight_decay}'] = dict()
        for fold in range(num_folds):
            print(f'Training model on fold {fold}; weight_decay: {weight_decay}', end='\n')

            # Run one training instance
            res_dict, model = run_mamba(df=df, cv_dict=cv_dict, fold=fold, weight_decay=weight_decay)
            print(' ')

            # Save model
            torch.save(model.state_dict(), f'../models/mamba/weight_decay/mamba_fold{fold}_weight_decay{weight_decay}.pt')

            # Save results on every loop in case early stop
            train_results[f'weight_decay_{weight_decay}'][f'fold_{fold}'] = res_dict
            #Save metadata
            output = open(f'../data/mamba_weight_decay_sweep_results.pkl', 'wb')
            pickle.dump(train_results, output)
            output.close()

def hidden_size_sweep(df, cv_dict, num_folds):
    # Sweep through learning rate values
    hidden_size_values = [1, 5, 10, 50, 100, 500, 1000]
    # hidden_size_values = [1000]

    train_results = dict()
    train_results['hidden_size_values'] = hidden_size_values

    for hidden_size in hidden_size_values:
        train_results[f'hidden_size_{hidden_size}'] = dict()
        for fold in range(num_folds):
            print(f'Training model on fold {fold}; hidden_size: {hidden_size}', end='\n')

            # Run one training instance
            res_dict, model = run_mamba(df=df, cv_dict=cv_dict, fold=fold, hidden_size=hidden_size)
            print(' ')

            # Save model
            torch.save(model.state_dict(), f'../models/mamba/hidden_size/mamba_fold{fold}_hidden_size{hidden_size}.pt')

            # Save results on every loop in case early stop
            train_results[f'hidden_size_{hidden_size}'][f'fold_{fold}'] = res_dict
            #Save metadata
            output = open(f'../data/mamba_hidden_size_sweep_results.pkl', 'wb')
            pickle.dump(train_results, output)
            output.close()


def main():
    df = pd.read_pickle('../data/developmental_df.pkl')
    n_subjects = len(np.unique(df['subj'].values))

    num_folds = 5
    cv_split = ShuffleSplit(n_splits=num_folds, test_size=.25, random_state=0)
    val_split = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)
    cv_dict = {}
    for fold, (train_val_idx, test_idx) in enumerate(cv_split.split(np.arange(n_subjects))):
        for t_idx, v_idx in val_split.split(train_val_idx): #No looping, just used to split train/validation sets
            cv_dict[fold] = {'train_idx':train_val_idx[t_idx], 
                            'test_idx':test_idx, 
                            'validation_idx':train_val_idx[v_idx]} 

    print(f'{n_subjects} unique subjects found')

    # bottleneck_sweep(df, cv_dict, num_folds)
    # learning_rate_sweep(df, cv_dict, num_folds)
    # weight_decay_sweep(df, cv_dict, num_folds)
    hidden_size_sweep(df, cv_dict, num_folds)

if __name__ == '__main__':
    main()

