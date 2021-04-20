import sys
import numpy as np

def run_combine_diff(diff_cm_mtx, emotions):
    n,m = diff_cm_mtx.shape
    assert n == m

    dir_cm_mtx = np.zeros((n, m))
    for i in range(n):
        for j in range(i+1, n):
            A_B = diff_cm_mtx[i][j]
            B_A = diff_cm_mtx[j][i]

            new = A_B - B_A
            if new > 0:
                dir_cm_mtx[i][j] = 1
            elif new < 0:
                dir_cm_mtx[j][i] = 1

            diff_cm_mtx[i][j] = abs(new)
            diff_cm_mtx[j][i] = abs(new)
    return diff_cm_mtx, dir_cm_mtx


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

    n,m = diff.shape

    assert n == m
    for i in range(n):
        for j in range(i+1, n):
            A_B = diff[i][j]
            B_A = diff[j][i]

            if A_B * B_A < 0:
                new = abs(A_B) + abs(B_A)
            else:
                new = max(abs(A_B), abs(B_A)) - min(abs(A_B), abs(B_A))

            diff[i][j] = new/2
            diff[j][i] = new/2
    np.savetxt(output_name, diff, delimiter=',')
