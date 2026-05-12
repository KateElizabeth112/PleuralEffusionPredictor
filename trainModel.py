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
import tomli


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
config_file = os.path.join(code_dir, "config.toml")


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

def getTestIDs(data, n_samples=1000):
    # get the IDs of the test data from the CheXpert dataset
    ids = random.sample(range(0, len(data)), 1000)

    return ids


def main():
    # load the CheXpert dataset
    train_dataset = CheXpertDataset(os.path.join(data_dir, "CheXpertSmall"), split='train', resized=True, transform=transforms.ToTensor())
    valid_dataset = CheXpertDataset(os.path.join(data_dir, "CheXpertSmall"), split='valid', resized=True, transform=transforms.ToTensor())

    # open the config file and load the parameters for the dataset
    with open(config_file, "rb") as f:
        config = tomli.load(f)

    train_dataset_size = config["data"]["train_dataset_size"]
    test_dataset_size = config["data"]["test_dataset_size"]
    #valid_dataset_size = config["data"]["valid_dataset_size"]

    # select a subset of the data to train the ResNet classifier on
    print(f"Selecting a subset of the data to train the ResNet classifier on with size {train_dataset_size}...")
    ids = getTrainIDs(n_samples=train_dataset_size)
    train_data = Subset(train_dataset, ids)

    # select a subset of the data to validate the ResNet classifier on
    #print(f"Selecting a subset of the data to validate the ResNet classifier on with size {valid_dataset_size}...")
    #validation_ids = getValidationIDs(dataset, n_samples=valid_dataset_size)
    #validation_ids = getTrainIDs(n_samples=valid_dataset_size)
    #validation_data = Subset(dataset, validation_ids)

    # select a subset of the data to test the ResNet classifier on
    print(f"Selecting a subset of the data to test the ResNet classifier on with size {test_dataset_size}...")
    #test_ids = getValidationIDs(dataset, n_samples=test_dataset_size)
    test_ids = getTrainIDs(n_samples=test_dataset_size)
    test_data = Subset(train_dataset, test_ids)

    # train the ResNet classifier on the selected subset of data and log results in MLFlow
    metrics = runTraining(train_data,
                          valid_dataset,
                          test_data,
                          output_dir,
                          config_file)
    


if __name__ == "__main__":
    main()