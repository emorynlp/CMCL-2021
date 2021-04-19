set -e -x
modelpath=$1
outputdir=$2
surfix=".bert.csv"
taskname="EDQA"

gpu=2
num_hidden_prob=8
modelname='MHP'
dim_mlps='128_64'
dims_hidden_prob=32
dropout=0.0

python finetune_classifier.py --gpu ${gpu} \
         --task_name ${taskname} \
         --epochs 1 \
     	 --modelname ${modelname} \
         --only_inference \
         --dev_batch_size 32 \
         --output_dir $outputdir \
    	 --surfix ${surfix} \
    	--num_hidden_prob ${num_hidden_prob} \
    	--dims_hidden_prob ${dims_hidden_prob} \
    	--dim_mlps ${dim_mlps} \
        --output_emb True \
	    --model_parameters ${modelpath}


