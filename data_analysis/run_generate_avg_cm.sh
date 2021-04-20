
y_true='data/valid.bert.csv'
path_prefix='emb_run'

for run in 1 2 3;
do
	## 1. Compute confusion matrices
	# Usage: python confusion_matrix.py [y_true_file] [y_pred_file]
	# The path of output file is hard coded as [y_pred_file].cm.npy
	for i in 32 64 128 enc; do
		y_pred=${path_prefix}${run}_valid/cls_res.${i}.emb.tsv
		python confusion_matrix.py ${y_true} ${y_pred}
	done
done


output_path='emb_run_valid_avg'
if [ ! -d ${output_path} ];
then
	mkdir ${output_path}
fi
for i in 32 64 128 enc; do
	python avg_cm_3layer.py ${path_prefix} ${i} ${output_path}/cls_res.${i}.emb.tsv.cm.npy
done
