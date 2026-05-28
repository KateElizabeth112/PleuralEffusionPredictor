# script for selecting a dataset from CheXpert, sampling a subset of the data, and training a ResNet classifier on it. Also calculates diversity metrics for the training data and logs everything in MLFlow.
import argparse
import mlflow
import os
from torch.utils.data import Subset
import numpy as np
from resnetTrainUtils import runTraining
import torchvision.transforms as transforms
import random
from cheXpertDataset import CheXpertDataset
import pandas as pd
import tomli
import pickle as pkl


# set up the argument parser
parser = argparse.ArgumentParser(description="Run experiments to determine the relationship between dataset diversity and generalisation performance")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located",
                    default="/Users/katephd/Documents/")
parser.add_argument("-c", "--config_file", type=str, help="Name of teh config file with the parameters for the dataset and training", default="config.toml")
parser.add_argument("-d", "--diversity", type=str, help="Whether to select the training data with high or low diversity score", default="high")

args = parser.parse_args()

# parse arguments
root_dir = args.root_dir
config_file = args.config_file
diversity = args.diversity

# set up paths to directories
code_dir = os.path.join(root_dir, "code/PleuralEffusionPredictor")
data_dir = os.path.join(root_dir, "data")
output_dir = os.path.join(root_dir, "output")
loss_plot_save_path = os.path.join(code_dir, "loss.png")
config_file_path = os.path.join(code_dir, config_file)


def sampleTrainIDs(n_samples=1000):
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


def getTestIDs(test_ids_file):
    # get the IDs of the training data from the CheXpert dataset by loading them from a file
    with open(test_ids_file, 'rb') as f:
        ids = pkl.load(f)

    return ids


def getTrainIDs(train_ids_file, diversity="high"):
    # get the IDs of the test data from the CheXpert dataset by loading them from a file
    with open(train_ids_file, 'rb') as f:
        ids_list = pkl.load(f)

    # pick the train IDs with the highest diversity score if diversity is "high", and pick the train IDs with the lowest diversity score if diversity is "low"
    diversity_scores = [item['diversity_score'] for item in ids_list]

    if diversity == "high":
        # find the index of the train IDs with the highest diversity score
        ids = ids_list[np.argmax(diversity_scores)]['train_ids']
    elif diversity == "low":
        # find the index of the train IDs with the lowest diversity score
        ids = ids_list[np.argmin(diversity_scores)]['train_ids']
    else:
        raise ValueError('diversity must be either "high" or "low"')

    return ids


def main():
    # open the config file and load the parameters for the dataset
    with open(config_file_path, "rb") as f:
        config = tomli.load(f)

    # load the parameters for the dataset from the config file
    train_dataset_size = config["data"]["train_dataset_size"]
    test_dataset_size = config["data"]["test_dataset_size"]
    test_ids_file = config["data"]["test_ids_file"]
    train_ids_file = config["data"]["train_ids_file"]

    # load the CheXpert dataset
    train_dataset = CheXpertDataset(os.path.join(data_dir, "CheXpertSmall"), split='train', resized=True, transform=transforms.ToTensor())
    valid_dataset = CheXpertDataset(os.path.join(data_dir, "CheXpertSmall"), split='valid', resized=True, transform=transforms.ToTensor())
    
    # select a subset of the data to train the ResNet classifier on
    print(f"Selecting a subset of the data to train the ResNet classifier on with size {train_dataset_size}...")
    train_ids = getTrainIDs(train_ids_file, diversity=diversity)
    train_data = Subset(train_dataset, train_ids)

    # select a subset of the data to test the ResNet classifier on
    print(f"Selecting a subset of the data to test the ResNet classifier on with size {test_dataset_size}...")
    test_ids = getTestIDs(test_ids_file)
    test_data = Subset(train_dataset, test_ids)

    # train the ResNet classifier on the selected subset of data and log results in MLFlow
    metrics = runTraining(train_data,
                          valid_dataset,
                          test_data,
                          output_dir,
                          config_file_path)
    


if __name__ == "__main__":
    main()