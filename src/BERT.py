from transformers import AutoModel
import pdb
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import json

from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import pipeline

from transformers import BertForSequenceClassification, AdamW, BertConfig

from HateSpeechBERT import HateSpeechBERT

def get_lr(optim):
    """ extracts the learning rate from the optimizer
    """
    for param_group in optim.param_groups:
        return param_group['lr']


def compute_loss(batch, model, criterion, phase, optim, device):
    """ gets the loss with or without a backprop whether the phase it's train or validation
    """
    input_ids, attention_mask, token_type_ids, labels = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    labels = labels.to(device)

    optim.zero_grad()
    with torch.set_grad_enabled(phase == 'train'):
        output = model(input_ids, attention_mask, token_type_ids)
        loss = criterion(output, labels)

        if phase == 'train':
            loss.backward()
            optim.step()
    return loss


def training(dataloaders, model, criterion, optim, lr_scheduler, epochs, device, best_loss):#, exp_dir):
    """ performs the full training process with printing results and advancing with the lr_scheduler
    the try catch block saves the training scores inside the epochs, altough interupted
    """
    print_frequency = 5

    #tensorboard_dir = os.path.join(exp_dir, 'runs/')
    #writer = SummaryWriter(tensorboard_dir)
    #print(f'Tensorboard is recording into folder: {tensorboard_dir}\n')
    try:
        results = []
        for epoch in range(epochs):
            for phase in ['train', 'test']:
                epoch_losses = []

                if phase == 'train':
                    # training mode
                    model.train()
                else:
                    # evaluate mode
                    model.eval()

                for i, batch in enumerate(dataloaders[phase]):
                    loss = compute_loss(batch, model, criterion, phase, optim,
                                        device)

                    epoch_losses.append(loss.item())
                    average_loss = np.mean(epoch_losses)
                    lr = get_lr(optim)

                    if (i + 1) % print_frequency == 0:
                        loading_percentage = int(
                            100 * (i + 1) / len(dataloaders[phase]))
                        print(
                            f'{phase}ing epoch {epoch}, iter = {i + 1}/{len(dataloaders[phase])} ' + \
                            f'({loading_percentage}%), loss = {loss}, average_loss = {average_loss} ' + \
                            f'learning rate = {lr}', end='\r')

                if phase == 'test' and average_loss < best_loss:
                    best_loss = average_loss

                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optim.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'validation_loss': best_loss
                    }, os.path.join("checkpoints", 'best_bert.pth'))

                if phase == 'train':
                    metric_results = {
                        'train_loss': average_loss
                    }

                if phase == 'test':
                    val_results = {
                        'val_loss': average_loss
                    }

                    metric_results.update(val_results)
                    results.append(metric_results)

                    try:
                        lr_scheduler.step()
                    except:
                        lr_scheduler.step(average_loss)

                print()
    except Exception as ex:
        print(ex)
    finally:
        results = pd.DataFrame(results)
        history_path = os.path.join("checkpoints", 'history.csv')
        results.to_csv(history_path)
        return best_loss


def get_exp_dir(config):
    """ returns a new folder to export the model weigths and configuration
    """
    exp_dir = f'data/logs/ro_bert_{config["seq_len"]}_{config["batch_size"]}'

    if config['fine_tune']:
        exp_dir += '_fine_tune'

    os.makedirs(exp_dir, exist_ok=True)

    experiments = [d for d in os.listdir(exp_dir) if
                   os.path.isdir(os.path.join(exp_dir, d))]
    experiments = set(map(int, experiments))
    if len(experiments) > 0:
        possible_experiments = set(range(1, max(experiments) + 2))
        experiment_id = min(possible_experiments - experiments)
    else:
        experiment_id = 1

    exp_dir = os.path.join(exp_dir, str(experiment_id))
    os.makedirs(exp_dir, exist_ok=True)

    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as fout:
        json.dump(config, fout)

    exp_weights = os.path.join(exp_dir, 'best.pth')
    config["experiment_weights"] = exp_weights
    return exp_dir


def load_weights(model, path, optimizer=None, lr_scheduler=None):
    """ loads the customized dict of weights
    """
    sd = torch.load(path)
    model.load_state_dict(sd['model'])
    if optimizer:
        optimizer.load_state_dict(sd['optimizer'])
    if lr_scheduler:
        lr_scheduler.load_state_dict(sd['lr_scheduler'])
    epoch = sd['epoch']
    validation_loss = sd['validation_loss']

    print(
        f'Loaded model from epoch {epoch + 1} with validation loss = {validation_loss} \n')
    return validation_loss


def train_bert(train_dataset, test_dataset):
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = HateSpeechBERT(n_sentiments=2, seq_len=2).to(device)

    criterion = nn.CrossEntropyLoss()

    training_dataset = train_dataset
    training_dataloader = DataLoader(training_dataset,  batch_size=32, shuffle=True)

    print(len(training_dataloader))
    for batch in training_dataloader:
        print(batch)
    testing_dataset = test_dataset
    testing_dataloader = DataLoader(testing_dataset, batch_size=32, shuffle=True)

    dataloaders = {
        'train': training_dataloader,
        'test': testing_dataloader
    }

    optim = Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = ReduceLROnPlateau(optim,
                                     factor=0.1,
                                     patience=3,
                                     mode='min')
    # if config["fine_tune"]:
    # load_weights(model) #, config["weights_path"])

    best_loss = 1e9
    # exp_dir = get_exp_dir(config)

    best_loss = training(dataloaders, model, criterion, optim, lr_scheduler, epochs=5, device=device, best_loss=best_loss)

if __name__ == "__main__":
    main_bert()
