import sys



def main(label_file, pred_file):

    labels1 = []
    with open(label_file, 'r') as fr:
        for line in fr.readlines():
            fs = line.strip().split('\t')
            labels1.append(fs[1])

    preds1 = []
    with open(pred_file, 'r') as fr:
        for line in fr.readlines():
            fs = line.strip().split('\t')
            preds1.append(fs[1])

    assert len(labels1) == len(preds1), 'label:{}, pred:{}'.format(len(labels1), len(preds1))

    acc1 = [ labels1[i] == preds1[i] for i in range(len(labels1))]
    acc1 = sum(acc1)/len(acc1)

    #from sklearn.metrics import classification_report
    #print(classification_report(labels1, preds1, target_names=['A', 'C', 'M', 'U']))


    return('acc1:{0:.3f}'.format(acc1))


if __name__ == '__main__':
    label_file = sys.argv[1]
    pred_file = sys.argv[2]
    print(main(label_file, pred_file))
