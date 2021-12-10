import matplotlib.pyplot as plt
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


def plot_shap(model, x_reshaped, elements=7, population=100, labels=None):
    background = x_reshaped[choice(x_reshaped.shape[0], population, replace=False)]
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(x_reshaped[1:elements])
    shap.image_plot(shap_values, -x_reshaped[1:elements], labels=labels)
