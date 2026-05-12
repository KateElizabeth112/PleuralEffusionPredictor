import os
import warnings
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

Metrics = namedtuple("Metrics", ["AUC", "ACC"])


class Evaluator:
    def __init__(self, task, split, image_size=None, root=None):
        self.split = split
        self.task = task

        if (image_size is None) or (image_size == 28):
            self.size = 28
            self.size_flag = ""
        else:
            self.size = image_size
            self.size_flag = f"_{image_size}"

        if root is not None and os.path.exists(root):
            self.root = root
        else:
            raise RuntimeError(
                "Failed to setup the default `root` directory. "
                + "Please specify and create the `root` directory manually."
            )

    def evaluate(self, y_score, y_targets, save_folder=None, run=None):
        # convert targets to uint8. Check for any invalid values in the targets before conversion and throw an error if there are any
        if np.any((y_targets < 0) | (y_targets > 1)):
            raise ValueError("Targets should be binary (0 or 1) for multi-label and binary-class tasks, and should be integers from 0 to n_classes-1 for multi-class tasks.")

        assert y_score.shape[0] == y_targets.shape[0]

        auc = getAUC(y_targets, y_score, self.task)
        acc = getACC(y_targets, y_score, self.task)
        metrics = Metrics(auc, acc)

        return metrics


def getAUC(y_true, y_score, task):
    """AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        auc = 0
        counter = 0
        for i in range(y_score.shape[1]):
            # check if our batch of samples has only one label, and if so, skip AUC calculation
            if np.unique(y_true[:, i]).shape[0] > 1:
                label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
                auc += label_auc
                counter += 1
        ret = auc / counter
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        if y_true.ndim == 2:
            y_true = y_true[:, -1]
        else:
            assert y_true.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            auc += roc_auc_score(y_true_binary, y_score_binary)
        ret = auc / y_score.shape[1]

    return ret


def getACC(y_true, y_score, task, threshold=0.5):
    """Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_score(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_score(y_true, y_score > threshold)
    else:
        ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret


def save_results(y_true, y_score, outputpath):
    """Save ground truth and scores
    :param y_true: the ground truth labels, shape: (n_samples, n_classes) for multi-label, and (n_samples,) for other tasks
    :param y_score: the predicted score of each class, shape: (n_samples, n_classes)
    :param outputpath: path to save the result csv

    """

    warnings.DeprecationWarning(
        "Only kept for backward compatiblility."
        + "Please use `Evaluator` API instead. "
    )
    idx = []

    idx.append("id")

    for i in range(y_true.shape[1]):
        idx.append("true_%s" % (i))
    for i in range(y_score.shape[1]):
        idx.append("score_%s" % (i))

    df = pd.DataFrame(columns=idx)
    for id in range(y_score.shape[0]):
        dic = {}
        dic["id"] = id
        for i in range(y_true.shape[1]):
            dic["true_%s" % (i)] = y_true[id][i]
        for i in range(y_score.shape[1]):
            dic["score_%s" % (i)] = y_score[id][i]

        df_insert = pd.DataFrame(dic, index=[0])
        df = df.append(df_insert, ignore_index=True)

    df.to_csv(outputpath, sep=",", index=False, header=True, encoding="utf_8_sig")
