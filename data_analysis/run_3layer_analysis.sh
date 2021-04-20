set -e -x

path='emb_run_valid_avg'

## 1. Compute the difference between confusion matrices
#  Usgae:  python compare_cm.py [input matrix 1] [input matrix 2] [output file]
enc_cm="cls_res.enc.emb.tsv.cm.npy"
ft0_cm="cls_res.128.emb.tsv.cm.npy"
ft1_cm="cls_res.64.emb.tsv.cm.npy"
ft2_cm="cls_res.32.emb.tsv.cm.npy"
diff_enc_ft0="diff_enc_ft0_3layer"
diff_ft0_ft1="diff_ft0_ft1_3layer"
diff_ft1_ft2="diff_ft1_ft2_3layer"
python compare_cm.py ${path}/${enc_cm} ${path}/${ft0_cm} ${path}/${diff_enc_ft0}
python compare_cm.py ${path}/${ft0_cm} ${path}/${ft1_cm} ${path}/${diff_ft0_ft1}
python compare_cm.py ${path}/${ft1_cm} ${path}/${ft2_cm} ${path}/${diff_ft1_ft2}

## 3. Compute on the difference matrices and visualize
python build_graph_3_combine.py ${path}/${diff_enc_ft0} ${path}/${diff_ft0_ft1} ${path}/${diff_ft1_ft2} \
2.0 ${path}/dot_3layer

