import warnings

import matplotlib.pyplot as plt
import numpy as np
import shap
from numpy import sum, unique, empty_like
from numpy.random import choice
from pandas import DataFrame
from seaborn import heatmap
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, figsize=(10, 10)):
    """
    Plots confusion matrix containing percentage
    """
    cm = confusion_matrix(y_true, y_pred, labels=unique(y_true))
    cm_sum = sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = DataFrame(cm, index=unique(y_true), columns=unique(y_true))
    cm.index.name = 'Predicted'
    cm.columns.name = 'Actual'
    fig, ax = plt.subplots(figsize=figsize)
    heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)


def random_sample(arr: np.array, size: int = 1) -> np.array:
    return arr[np.random.choice(len(arr), size=size, replace=False)]


def plot_shap(model, data, background=[], labels=None, file=None):
    """
    Generates shap plot based on given model and input. If file is provided then file is persisted.
    """

    if len(background) == 0:
        background = data

    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(data)

    shap.image_plot(shap_values, -data, labels=labels, show=file is None)

    if file:
        plt.savefig(file)

def first_occurence(data):
    """
    Returns the array indexes of first occurences of each digit. I.E. it returns at which position the first 0 occurecs,
    followed by the first 1, 2, 3, 4, 5, 6, 7, 8, 9.

    :param data:
    :return:
    """
    return np.array([
        data[data == 0].first_valid_index(),
        data[data == 1].first_valid_index(),
        data[data == 2].first_valid_index(),
        data[data == 3].first_valid_index(),
        data[data == 4].first_valid_index(),
        data[data == 5].first_valid_index(),
        data[data == 6].first_valid_index(),
        data[data == 7].first_valid_index(),
        data[data == 8].first_valid_index(),
        data[data == 9].first_valid_index(),
    ])
