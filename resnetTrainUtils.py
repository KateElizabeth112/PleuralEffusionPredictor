import argparse
import os
import time
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Subset
import torchvision.transforms as transforms
from evaluatorLocal import Evaluator
from models import ResNet18, ResNet50
from torchvision.models import resnet18, resnet50
from tqdm import trange
import tomli


def runTraining(train_dataset, val_dataset, test_dataset, output_dir, config_file):
    # load the toml config file with the parameters for training the ResNet classifier
    with open(config_file, "rb") as f:
        config = tomli.load(f)

    # load the training parameters
    lr = config["training"]["lr"]
    gamma = config["training"]["gamma"]
    num_epochs = config["training"]["num_epochs"]
    milestones = [0.5 * num_epochs, 0.75 * num_epochs]
    batch_size = config["training"]["batch_size"]

    # load the model parameters
    task = config["model"]["task"]              # type of classification task e.g. binary-class, multi-label binary-class or multi-class
    n_channels = config["model"]["n_channels"]  # number of input channels for the ResNet
    n_classes = config["model"]["n_classes"]    # number of classes for the ResNet output
    model_type = config["model"]["model_type"]  # type of ResNet to use (e.g. resnet18 or resnet50)
    image_size = config["data"]["image_size"]  # size of the input images (assumed to be square)

    # set the device to GPU if available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # create a folder to save the model and results, named as the current time
    results_dir = os.path.join(output_dir, time.strftime("%y%m%d_%H%M%S"))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print('==> Preparing data...')

    # prepare data loaders for training, validation and testing
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True)
    train_loader_at_eval = data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False)

    print('==> Building and training model for {} epochs...'.format(num_epochs))

    if model_type == 'resnet18':
        model = ResNet18(in_channels=n_channels, num_classes=n_classes)
    elif model_type == 'resnet50':
        model = ResNet50(in_channels=n_channels, num_classes=n_classes)
    else:
        raise NotImplementedError

    model = model.to(device)

    train_evaluator = Evaluator(task, 'train', image_size=image_size, root=results_dir)
    val_evaluator = Evaluator(task, 'val', image_size=image_size, root=results_dir)
    test_evaluator = Evaluator(task, 'test', image_size=image_size, root=results_dir)

    if task == "multi-label, binary-class" or task == "binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if num_epochs == 0:
        return

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    logs = ['loss', 'auc', 'acc']
    train_logs = ['train_' + log for log in logs]
    val_logs = ['val_' + log for log in logs]
    test_logs = ['test_' + log for log in logs]
    log_dict = OrderedDict.fromkeys(train_logs + val_logs + test_logs, 0)

    best_auc = 0
    best_epoch = 0
    best_model = deepcopy(model)
    train_loss = []

    global iteration
    iteration = 0

    for epoch in trange(num_epochs):
        train_loss.append(train(model, train_loader, task, criterion, optimizer, device))

        train_metrics = test(model, train_evaluator, train_loader_at_eval, task, criterion, device)
        val_metrics = test(model, val_evaluator, val_loader, task, criterion, device)
        test_metrics = test(model, test_evaluator, test_loader, task, criterion, device)

        scheduler.step()

        for i, key in enumerate(train_logs):
            log_dict[key] = train_metrics[i]
        for i, key in enumerate(val_logs):
            log_dict[key] = val_metrics[i]
        for i, key in enumerate(test_logs):
            log_dict[key] = test_metrics[i]

        cur_auc = val_metrics[1]
        if cur_auc > best_auc:
            best_epoch = epoch
            best_auc = cur_auc
            best_model = deepcopy(model)
            print('cur_best_auc:', best_auc)
            print('cur_best_epoch', best_epoch)

    state = {
        'net': best_model.state_dict(),
    }

    path = os.path.join(results_dir, 'best_model.pth')
    torch.save(state, path)

    train_metrics = test(best_model, train_evaluator, train_loader_at_eval, task, criterion, device)
    val_metrics = test(best_model, val_evaluator, val_loader, task, criterion, device)
    test_metrics = test(best_model, test_evaluator, test_loader, task, criterion, device)

    train_log = 'train  auc: %.5f  acc: %.5f\n' % (train_metrics[1], train_metrics[2])
    val_log = 'val  auc: %.5f  acc: %.5f\n' % (val_metrics[1], val_metrics[2])
    test_log = 'test  auc: %.5f  acc: %.5f\n' % (test_metrics[1], test_metrics[2])

    log = '%s\n' %  train_log + val_log + test_log
    print(log)

    metrics = {"train_AUC": train_metrics[1],
               "train_acc": train_metrics[2],
               "val_AUC": val_metrics[1],
               "val_acc": val_metrics[2],
               "test_AUC": test_metrics[1],
               "test_acc": test_metrics[2]}

    return metrics


def train(model, train_loader, task, criterion, optimizer, device):
    total_loss = []
    global iteration

    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        if task == 'multi-label, binary-class' or task == 'binary-class':
            targets = targets.to(torch.float32).to(device)
            loss = criterion(outputs, targets)
        else:
            targets = torch.squeeze(targets, 1).long().to(device)
            loss = criterion(outputs, targets)

        total_loss.append(loss.item())

        if iteration % 10 == 0:
            print('train_loss {0:.3f}, iter: {1}'.format(loss.item(), iteration))
        iteration += 1

        loss.backward()
        optimizer.step()

    epoch_loss = sum(total_loss) / len(total_loss)
    return epoch_loss


def test(model, evaluator, data_loader, task, criterion, device, run=None, save_folder=None):
    model.eval()

    total_loss = []
    y_score = torch.tensor([]).to(device)
    y_targets = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if task == 'multi-label, binary-class' or task == 'binary-class':
                targets = targets.to(torch.float32).to(device)
                loss = criterion(outputs, targets)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = torch.squeeze(targets, 1).long().to(device)
                loss = criterion(outputs, targets)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            total_loss.append(loss.item())
            y_score = torch.cat((y_score, outputs), 0)
            y_targets = torch.cat((y_targets, targets))

        y_score = y_score.detach().cpu().numpy()
        y_targets = y_targets.detach().cpu().numpy()
        auc, acc = evaluator.evaluate(y_score, y_targets)

        test_loss = sum(total_loss) / len(total_loss)

        return [test_loss, auc, acc]


  