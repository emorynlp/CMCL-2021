# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-10-24 12:53
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          figname='confusion_matrix.png',
                          cm_name='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        #cm = np.ceil(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    if len(cm_name) != 0:
        np.save(cm_name, cm)

    figsize = 12
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation='vertical')

    # Loop over data dimensions and create text annotations.
    fmt = '.0f' if normalize else 'd'
    #fmt = '.1f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(figname)
    return ax, cm


if __name__ == '__main__':

    #np.set_printoptions(precision=3)

    assert len(sys.argv) >= 3, 'usage: python confusion_matrix.py [y_true_file] [y_pred_file]'
    y_true_file = sys.argv[1] # 2nd col is label 
    y_pred_file = sys.argv[2] # 2nd col is label
    y_true = []
    y_pred = []

    with open(y_true_file, 'r') as fr:
        for line in fr.readlines():
            line = line.strip()
            fs = line.split('\t')
            assert len(fs) >= 2
            y_true.append(fs[1])

    with open(y_pred_file, 'r') as fr:
        for line in fr.readlines():
            line = line.strip()
            fs = line.split('\t')
            assert len(fs) >= 2
            y_pred.append(fs[1])

    emotions = []
    with open('emotions.txt', 'r') as fr:
        emotions = [x.strip() for x in fr.readlines()]

    le = preprocessing.LabelEncoder()
    le.fit(emotions)

    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)

    _, cm = plot_confusion_matrix(y_true, y_pred, classes=le.classes_, normalize=True, figname=y_pred_file+'.pdf', cm_name=y_pred_file+'.cm.npy')

