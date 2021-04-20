import sys
import numpy as np

if __name__ == '__main__':
    cm1 = sys.argv[1]
    cm2 = sys.argv[2]
    output_name = sys.argv[3]

    cm1_mtx = np.load(cm1)
    cm2_mtx = np.load(cm2)

    diff = cm2_mtx - cm1_mtx
    emotions = []
    with open('emotions_sorted.txt', 'r') as fr:
        emotions = [x.strip() for x in fr.readlines()]

    np.savetxt(output_name, diff, delimiter=',')
