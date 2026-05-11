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
import pandas as pd


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
image_size = 320


def getTrainIDs(n_samples=1000):
    # get the IDs of the training data from the CheXpert dataset

    # First return the IDs of the images that have a pleural effusion label of 1 OR 0, then return a random sample of the remaining IDs to make up a total of n_samples
    # ignore -1 or NaN labels for pleural effusion
    df = pd.read_csv(os.path.join(data_dir, "CheXpertSmall", "train_reduced.csv"))
    pleural_effusion_ids = df[(df['Pleural Effusion'] == 1.0) | (df['Pleural Effusion'] == 0.0)]['image_id'].values

    # select a random sample of the pleural effusion ids to make up a total of n_samples
    if len(pleural_effusion_ids) >= n_samples:
        ids = np.array(random.sample(list(pleural_effusion_ids), n_samples))
    else:
        # throw a warning if there are not enough pleural effusion ids to make up n_samples
        print(f"Warning: there are only {len(pleural_effusion_ids)} pleural effusion ids in the dataset, which is less than the requested n_samples of {n_samples}. Returning all pleural effusion ids.")
        ids = pleural_effusion_ids
    
    return ids


def getTrainIDs(data, n_samples=1000):
    # get the IDs of the validation data from the CheXpert dataset
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
    dataset = CheXpertDataset(os.path.join(data_dir, "CheXpertSmall"), split='train', resized=True, transform=transforms.ToTensor())


    # select a subset of the data to train the ResNet classifier on
    ids = getTrainIDs(dataset)
    train_data = Subset(dataset, ids)

    # select a subset of the data to validate the ResNet classifier on
    validation_ids = getValidationIDs(dataset)
    validation_data = Subset(dataset, validation_ids)

    # select a subset of the data to test the ResNet classifier on
    test_ids = getValidationIDs(dataset)
    test_data = Subset(dataset, test_ids)


    # train the ResNet classifier on the selected subset of data and log results in MLFlow
    
    metrics = runTraining(train_data,
                          validation_data,
                          test_data,
                          output_dir,
                          n_epochs,
                          batch_size,
                          image_size,
                          'resnet50')
    


if __name__ == "__main__":
    main()