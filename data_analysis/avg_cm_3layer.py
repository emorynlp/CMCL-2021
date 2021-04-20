import sys
import numpy as np

if __name__ == '__main__':

    path_prefix = sys.argv[1]
    emb_ver = sys.argv[2]
    output_name = sys.argv[3]
    cm1 = '{}1_valid/cls_res.{}.emb.tsv.cm.npy'.format(path_prefix, emb_ver)
    cm2 = '{}2_valid/cls_res.{}.emb.tsv.cm.npy'.format(path_prefix, emb_ver)
    cm3 = '{}3_valid/cls_res.{}.emb.tsv.cm.npy'.format(path_prefix, emb_ver)

    cm1_mtx = np.load(cm1)
    cm2_mtx = np.load(cm2)
    cm3_mtx = np.load(cm3)

    avg_mtx = (cm1_mtx + cm2_mtx + cm3_mtx)/3
    np.save(output_name, avg_mtx)
