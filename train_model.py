import os
import random
import torch
import torch_geometric
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import pickle
from scipy import ndimage, stats
from scipy.spatial.distance import cdist
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import Sequential,GCNConv, SAGEConv, BatchNorm
from torch.nn import Linear, ReLU,CrossEntropyLoss


import os
import numpy as np
import time
import torch
import copy
import sklearn.metrics


import metadata

import csv
import os
import argparse
import time

import GPUtil
import sys
from numpy import unicode
from datetime import datetime


def write_header(output_filename):
    fieldnames = ['Epoch', 'Time stamp', 'ID', 'Name', 'Serial', 'UUID', 'GPU temp. [C]', 'GPU util. [%]', 'Memory util. [%]',
                  'Memory total [MB]', 'Memory used [MB]', 'Memory free [MB]', 'Display mode', 'Display active']
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()


def record(epoch, output_filename):
    #out_name, out_file_extension = os.path.splitext(output_filename)
    output_dir = os.path.dirname(output_filename)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    GPUs = GPUtil.getGPUs()
    print('INFO: GPUs:', GPUs)
    if len(GPUs) < 1:
        print('WARNING: the hardware does not contain NVIDIA GPU card')
        return

    now = datetime.now()  # current date and time
    date_time = now.strftime("%Y:%m:%d:%H:%M:%S")
    print('INFO: date_time ', date_time)

    attrList = [[{'attr': 'id', 'name': 'ID'},
                 {'attr': 'name', 'name': 'Name'},
                 {'attr': 'serial', 'name': 'Serial'},
                 {'attr': 'uuid', 'name': 'UUID'}],
                [{'attr': 'temperature', 'name': 'GPU temp.', 'suffix': 'C', 'transform': lambda x: x, 'precision': 0},
                 {'attr': 'load', 'name': 'GPU util.', 'suffix': '%', 'transform': lambda x: x * 100, 'precision': 0},
                 {'attr': 'memoryUtil', 'name': 'Memory util.', 'suffix': '%', 'transform': lambda x: x * 100,
                  'precision': 0}],
                [{'attr': 'memoryTotal', 'name': 'Memory total', 'suffix': 'MB', 'precision': 0},
                 {'attr': 'memoryUsed', 'name': 'Memory used', 'suffix': 'MB', 'precision': 0},
                 {'attr': 'memoryFree', 'name': 'Memory free', 'suffix': 'MB', 'precision': 0}],
                [{'attr': 'display_mode', 'name': 'Display mode'},
                 {'attr': 'display_active', 'name': 'Display active'}]]


    # store the date_time as teh first entry in the recorded row
    store_gpu_info = str(epoch) + ',' + date_time

    for attrGroup in attrList:
        #print('INFO: attrGroup:', attrGroup)

        index = 1
        for attrDict in attrGroup:
            attrPrecision = '.' + str(attrDict['precision']) if ('precision' in attrDict.keys()) else ''
            attrTransform = attrDict['transform'] if ('transform' in attrDict.keys()) else lambda x: x

            for gpu in GPUs:
                attr = getattr(gpu, attrDict['attr'])

                attr = attrTransform(attr)

                if (isinstance(attr, float)):
                    attrStr = ('{0:' + attrPrecision + 'f}').format(attr)
                elif (isinstance(attr, int)):
                    attrStr = ('{0:d}').format(attr)
                elif (isinstance(attr, str)):
                    attrStr = attr;
                elif (sys.version_info[0] == 2):
                    if (isinstance(attr, unicode)):
                        attrStr = attr.encode('ascii', 'ignore')
                else:
                    raise TypeError(
                        'Unhandled object type (' + str(type(attr)) + ') for attribute \'' + attrDict['name'] + '\'')

                #print('INFO: attrStr ', attrStr)
                store_gpu_info += ',' + attrStr
                index +=1

    store_gpu_info += '\n'
    print('row data:', store_gpu_info)
    with open(output_filename, 'a', newline='') as csvfile:
        csvfile.write(store_gpu_info)

def compute_metrics(x, y, stats, name, epoch):
    # convert x to numpy
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    # convert x from one hot to class label
    x = np.argmax(x, axis=1)  # assumes NCHW tensor order

    assert x.shape == y.shape

    # flatten into a 1d vector
    x = x.flatten()
    y = y.flatten()

    val = sklearn.metrics.accuracy_score(y, x)
    stats.add(epoch, '{}_accuracy'.format(name), val)

    val = sklearn.metrics.f1_score(y, x, average = 'micro')
    stats.add(epoch, '{}_f1'.format(name), val)

    val = sklearn.metrics.jaccard_score(y, x, average = 'micro')
    stats.add(epoch, '{}_jaccard'.format(name), val)

    val = sklearn.metrics.confusion_matrix(y, x)
    stats.add(epoch, '{}_confusion'.format(name), val)

    return stats


def eval_model(model, dataloader, criterion, device, epoch, stats, name):
    start_time = time.time()

    avg_loss = 0
    batch_count = len(dataloader)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            
            data_input = batch.to(device)
            pred = model(data_input)
            masks = data_input.y.to(torch.int64)
            

            

            # compute metrics
            batch_train_loss = criterion(pred, masks)
                


            avg_loss += batch_train_loss.item()

    avg_loss /= batch_count
    wall_time = time.time() - start_time

    stats.add(epoch, '{}_wall_time'.format(name), wall_time)
    stats.add(epoch, '{}_loss'.format(name), avg_loss)


def train_epoch(model, dataloader, optimizer, criterion, lr_scheduler, device, epoch, stats):
    avg_train_loss = 0

    model.train()
    batch_count = len(dataloader)


    start_time = time.time()
    epoch_loss = 0
    

    for batch in dataloader:
        optimizer.zero_grad()
        data_input = batch.to(device)
        pred = model(data_input)
        masks = data_input.y.to(torch.int64)
        

        #stats = compute_metrics(pred, masks, stats, name='train', epoch=epoch)

        # compute metrics
        batch_train_loss = criterion(pred, masks)
            


        batch_train_loss.backward()
        avg_train_loss += batch_train_loss.item()

        optimizer.step()


        if lr_scheduler is not None:
            lr_scheduler.step()

    avg_train_loss /= batch_count
    wall_time = time.time() - start_time


    stats.add(epoch, 'train_wall_time', wall_time)
    stats.add(epoch, 'train_loss', avg_train_loss)
    return model


def train(train_dataset, val_dataset, model, output_filepath, num_classes, learning_rate, early_stopping_epoch_count=5, loss_eps=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (device)
    
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    gpu_log_filename = os.path.join(output_filepath, "gpu_log.csv")
    write_header(gpu_log_filename)

    train_dl = DataLoader(train_dataset, batch_size=1, shuffle = True)
    val_dl = DataLoader(val_dataset, batch_size=1, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    class_count = np.zeros(num_classes)
    for data_item in train_dl:
        for i in range(num_classes):
            class_count[i] += np.sum(np.array(data_item.y) == i)
        
    loss_weight = torch.tensor(np.max(class_count)/class_count).to(torch.float)

    criterion = CrossEntropyLoss(weight = loss_weight.to(device))

    
    model = model.to(device)

    epoch = 0
    done = False
    best_model = model
    stats = metadata.TrainingStats()

    start_time = time.time()

    while epoch<1000:
        print('Epoch: {}'.format(epoch))

        model = train_epoch(model, train_dl, optimizer, criterion, lr_scheduler, device, epoch, stats)


        eval_model(model, val_dl, criterion, device, epoch, stats, 'val')
        record(epoch, gpu_log_filename)

        # handle recording the best model stopping
        val_loss = stats.get('{}_loss'.format('val'))
        error_from_best = np.abs(val_loss - np.min(val_loss))
        error_from_best[error_from_best < np.abs(loss_eps)] = 0
        # if this epoch is with convergence tolerance of the global best, save the weights
        if error_from_best[epoch] == 0:
            #print('Updating best model with epoch: {} loss: {}, as its less than the best loss plus eps {}.'.format(epoch, val_loss[epoch], loss_eps))
            best_model = copy.deepcopy(model)

            # update the global metrics with the best epoch
            stats.update_global(epoch)

        stats.add_global('training_wall_time', sum(stats.get('train_wall_time')))
        stats.add_global('val_wall_time', sum(stats.get('val_wall_time')))

        # update the number of epochs trained
        stats.add_global('num_epochs_trained', epoch)
        # write copy of current metadata metrics to disk
        stats.export(output_filepath)

        # handle early stopping
        best_val_loss_epoch = np.where(error_from_best == 0)[0][0]  # unpack numpy array, select first time since that value has happened
        if epoch >= (best_val_loss_epoch + early_stopping_epoch_count):
            print("Exiting training loop in epoch: {} - due to early stopping criterion being met".format(epoch))
            done = True
        epoch += 1

    if test_dataset is not None:
        print('Evaluating model against test dataset')
        eval_model(model, test_dl, criterion, device, epoch, stats, 'test')
        # update the global metrics with the best epoch, to include test stats
        stats.update_global(epoch)

    wall_time = time.time() - start_time
    stats.add_global('wall_time', wall_time)
    print("Total WallTime: ", stats.get_global('wall_time'), 'seconds')

    stats.export(output_filepath)  # update metrics data on disk
    best_model.cpu()  # move to cpu before saving to simplify loading the model
    torch.save(best_model.state_dict(), os.path.join(output_filepath, 'model.pt'))
