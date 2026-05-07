# script for selecting a dataset from CheXpert, sampling a subset of the data, and training a ResNet classifier on it. Also calculates diversity metrics for the training data and logs everything in MLFlow.
import argparse
import mlflow
import os
from torch.utils.data import Subset
import pickle as pkl
import numpy as np
from resnetTrainUtils import runTraining
import torchvision.transforms as transforms
import random
from cheXpertDataset import CheXpertDataset


# set up the argument parser
parser = argparse.ArgumentParser(description="Run experiments to determine the relationship between dataset diversity and generalisation performance")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located",
                    default="/Users/katephd/Documents/")

args = parser.parse_args()

# set up paths to directories
root_dir = args.root_dir
code_dir = os.path.join(root_dir, "code/PleuralEffusionPredictor")
data_dir = os.path.join(root_dir, "data")
output_dir = os.path.join(root_dir, "output")
loss_plot_save_path = os.path.join(code_dir, "loss.png")


# set up parameters for training the ResNet classifier
dataset_name = "CheXpert"
n_epochs = 3
batch_size = 5
image_size = 128


def getTrainIDs(data, n_samples=1000):
    # get the IDs of the training data from the CheXpert dataset
    ids = random.sample(range(0, len(data)), 1000)

    return ids

def getValidationIDs(data, n_samples=1000):
    # get the IDs of the validation data from the CheXpert dataset
    ids = random.sample(range(0, len(data)), 1000)

    return ids

def getTestIDs(data, n_samples=1000):
    # get the IDs of the test data from the CheXpert dataset
    ids = random.sample(range(0, len(data)), 1000)

    return ids


def main():
    # load the CheXpert dataset
    dataset = CheXpertDataset(os.path.join(data_dir, "CheXpertSmall"), split='train', transform=transforms.ToTensor())


    # select a subset of the data to train the ResNet classifier on
    ids = getTrainIDs()
    train_data = Subset(dataset, ids)

    # select a subset of the data to validate the ResNet classifier on
    validation_ids = getValidationIDs()
    validation_data = Subset(dataset, validation_ids)

    # select a subset of the data to test the ResNet classifier on
    test_ids = getValidationIDs()
    test_data = Subset(dataset, test_ids)


    # train the ResNet classifier on the selected subset of data and log results in MLFlow
    metrics = runTraining(train_data,
                          validation_data,
                          test_data,
                          dataset_name,
                          output_dir,
                          n_epochs,
                          batch_size,
                          image_size,
                          'resnet50',
                          True)




if __name__ == "__main__":
    main()