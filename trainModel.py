# script for selecting a dataset from CheXpert, sampling a subset of the data, and training a ResNet classifier on it. Also calculates diversity metrics for the training data and logs everything in MLFlow.
import argparse
import mlflow
import os
from torch.utils.data import Subset
import numpy as np
from resnetTrainUtils import runTraining
import torchvision.transforms as transforms
from cheXpertDataset import CheXpertDataset
import pandas as pd
import tomli
import pickle as pkl


# set up the argument parser
parser = argparse.ArgumentParser(description="Run experiments to determine the relationship between dataset diversity and generalisation performance")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located",
                    default="/Users/katephd/Documents/")
parser.add_argument("-c", "--config_file", type=str, help="Name of teh config file with the parameters for the dataset and training", default="config.toml")
parser.add_argument("-t", "--train_ids_file", type=str, help="Name of the pickle file containing the train IDs to use for trainingn", default="train_ids_1780652917.pkl")

args = parser.parse_args()

# parse arguments
root_dir = args.root_dir
config_file = args.config_file
train_ids_file = args.train_ids_file

# set up paths to directories
code_dir = os.path.join(root_dir, "code/PleuralEffusionPredictor")
data_dir = os.path.join(root_dir, "data")
output_dir = os.path.join(root_dir, "output")
loss_plot_save_path = os.path.join(code_dir, "loss.png")
config_file_path = os.path.join(code_dir, config_file)

# Point MLflow to the local tracking server
#mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create or use an experiment
mlflow.set_experiment("PleuralEffusionPredictor")


def getIDs(ids_file):
    # get the IDs of the data from the CheXpert dataset by loading them from a file
    with open(os.path.join("ids", ids_file), 'rb') as f:
        ids = pkl.load(f)

    return ids


def main():
    # open the config file and load the parameters for the dataset
    with open(config_file_path, "rb") as f:
        config = tomli.load(f)

    # load the parameters for the dataset from the config file
    test_ids_file = config["data"]["test_ids_file"]

    # use the train IDs file to retrieve the associated diversity score for the dataset
    diversityScores = pd.read_pickle("ids/diversity_scores.pkl")

    # turn into a dictionary where the key is the train IDs file and the value is the diversity score. 
    diversityScores = {x["filename"]: x["diversity_score"] for x in diversityScores}

    # retrieve the diversity score for the current train IDs file
    diversity_score = diversityScores.get(train_ids_file, None)

    # load the CheXpert dataset
    train_dataset = CheXpertDataset(os.path.join(data_dir, "CheXpertSmall"), split='train', resized=True, transform=transforms.ToTensor())
    valid_dataset = CheXpertDataset(os.path.join(data_dir, "CheXpertSmall"), split='valid', resized=True, transform=transforms.ToTensor())
    
    # select a subset of the data to train the ResNet classifier on
    train_ids = getIDs(train_ids_file)
    train_data = Subset(train_dataset, train_ids)

    # select a subset of the data to test the ResNet classifier on
    test_ids = getIDs(test_ids_file)
    test_data = Subset(train_dataset, test_ids)

    # train the ResNet classifier on the selected subset of data and log results in MLFlow
    metrics = runTraining(train_data,
                          valid_dataset,
                          test_data,
                          output_dir,
                          config_file_path)
    
    # record the parameters and metrics using MLFlow
    with mlflow.start_run():
        mlflow.log_params(config["training"])
        mlflow.log_params(config["data"])
        mlflow.log_params(config["model"])
        mlflow.log_param("train_data_size", len(train_data))
        mlflow.log_param("diversity_score", diversity_score)
        mlflow.log_param("train_ids_file", train_ids_file)
        mlflow.log_param("test_ids_file", test_ids_file)
        mlflow.log_metrics(metrics)
        mlflow.log_metric("test_AUC", metrics["test_AUC"])
        mlflow.log_metric("test_acc", metrics["test_acc"])



if __name__ == "__main__":
    main()