import numpy as np
import mxnet as mx
import sys
from scipy.special import logit

from data.classification import EDTask
task = EDTask('.bert.csv')
classes = task.class_labels


if __name__ == '__main__':
    emb_mtx_file = sys.argv[1]
    label_file = 'data/valid.bert.csv'

    emb_mtx = np.loadtxt(emb_mtx_file, delimiter='\t') # (n, 192)

    labels = []
    with open(label_file, 'r') as fr:
        for line in fr.readlines():
            fs = line.strip().split('\t')
            labels.append(fs[1])

    n, p = emb_mtx.shape
    assert len(labels) == n

    emo2mtx = {}
    for emotion in classes:
        emo2mtx[emotion] = []

    for i in range(n):
        label = labels[i]
        emo2mtx[label].append(emb_mtx[i].tolist())

    new_mtx = np.zeros((32, p))
    for i, emotion in enumerate(classes):
        mtx = np.asarray(emo2mtx[emotion])
        mtx = np.mean(mtx, axis=0)
        new_mtx[i] = mtx

    np.save(emb_mtx_file + '.emb', new_mtx)
