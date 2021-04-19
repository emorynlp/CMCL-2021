import numpy as np
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pdb

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
def get_label(label_file, test=False, cls=False):
    labels = []
    with open(label_file, 'r') as fr:
        for line in fr.readlines():
            fs = line.strip().split('\t')
            labels.append(fs[1])

    if cls:
        if not test:
            le.fit(labels)
        return le.transform(labels)

    return labels

def reverse_labels(preds):
    return le.inverse_transform(preds)

def run_cls(train_X, train_y, test_X, test_y):
    model = LogisticRegression(random_state=0, n_jobs=8, solver='lbfgs', max_iter=5000)
    model.fit(train_X, train_y)
    pred_y = model.predict(test_X)
    pred_y_proba = model.predict_proba(test_X)
    score_acc = model.score(test_X, test_y)

    score_auc = roc_auc_score(test_y, pred_y_proba, multi_class='ovr')

    transformed_preds = reverse_labels(pred_y)

    res_file = '{}_valid/cls_res.{}emb.score'.format(emb_path, surfix)
    with open(res_file, 'w') as fw:
        fw.write('auroc:{0:.3f}\t{1:.3f}\n'.format(score_auc, score_acc))

    output_file = '{}_valid/cls_res.{}emb.tsv'.format(emb_path, surfix)
    print('write results {}'.format(output_file))
    with open(output_file, 'w') as fw:
        fw.write('\n'.join(['{}\t{}'.format(i, pred) for i, pred in enumerate(transformed_preds)])  + '\n')


if __name__ == '__main__':
    dataname = 'EDQA'
    surfix = sys.argv[1]
    emb_path = sys.argv[2]

    test_X = np.loadtxt('{}_valid/{}_none.{}emb.tsv'.format(emb_path, dataname, surfix), delimiter='\t')
    train_X = np.loadtxt('{}_train/{}_none.{}emb.tsv'.format(emb_path, dataname, surfix), delimiter='\t')

    train_y = get_label('data/train.bert.csv', cls=True)
    test_y = get_label('data/valid.bert.csv', test=False, cls=True)

    run_cls(train_X, train_y, test_X, test_y)
