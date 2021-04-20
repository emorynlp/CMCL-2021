import sys

from graphviz import Digraph
from graph_utils import gen_graph
import numpy as np
from combine_diff import run_combine_diff

emotions = []
with open('emotions_sorted.txt', 'r') as fr:
    emotions = [x.strip() for x in fr.readlines()]


def get_diff(a, b):
    if a > b:
        return 1
    elif a == b:
        return 0
    else:
        return -1

def uniq(lst):
    new_lst = []
    for i in lst:
        if i not in new_lst:
            new_lst.append(i)
    return new_lst


def merge(pairs_0, pairs_1):
    res = [emo1 for _, emo1 in pairs_0]
    res += [emo2 for emo2, _ in pairs_1]
    res = uniq(res)
    return res


def get_pairs(cm_mtx, dir_mtx, threshold):
    pairs = []
    n, m = cm_mtx.shape
    for i in range(n):
        for j in range(n):
            if abs(cm_mtx[i][j]) >= threshold and dir_mtx[i][j] == 1:
                pairs.append((emotions[i], emotions[j]))

    pairs = uniq(pairs)
    return pairs


if __name__ == '__main__':
    cm1 = sys.argv[1]
    cm2 = sys.argv[2]
    cm3 = sys.argv[3]
    threshold = float(sys.argv[4])
    output_name = sys.argv[5]

    cm1_mtx = np.loadtxt(cm1, delimiter=',')
    cm2_mtx = np.loadtxt(cm2, delimiter=',')
    cm3_mtx = np.loadtxt(cm3, delimiter=',')

    combined_cm1_mtx, dir_cm1_mtx = run_combine_diff(cm1_mtx, emotions)
    combined_cm2_mtx, dir_cm2_mtx = run_combine_diff(cm2_mtx, emotions)
    combined_cm3_mtx, dir_cm3_mtx = run_combine_diff(cm3_mtx, emotions)
    print(dir_cm1_mtx)

    pairs_0 = get_pairs(combined_cm1_mtx, dir_cm1_mtx, threshold)
    pairs_1 = get_pairs(combined_cm2_mtx, dir_cm2_mtx, threshold)
    pairs_2 = get_pairs(combined_cm3_mtx, dir_cm3_mtx, threshold)
    print(pairs_0)


    color = 'black'
    reverse = False
    mode = 'combined'

    gen_graph(pairs_0, '{}_{}_diff_enc_layer0'.format(output_name, mode),  color, emotions, combined_cm1_mtx, reverse)
    gen_graph(pairs_1, '{}_{}_diff_layer0_layer1'.format(output_name, mode), color, emotions, combined_cm2_mtx, reverse)
    gen_graph(pairs_2, '{}_{}_diff_layer1_layer2'.format(output_name, mode), color, emotions, combined_cm3_mtx, reverse)
