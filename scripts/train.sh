#!/bin/bash
set -e -x

taskname=EDQA
surfix='.bert.csv'

gpu=0

# the number of feature vectors in each layer
num_hidden_prob=8

# date info, used for naming the output directory
dt=0609

# The model type can be 'baseline' or 'MHP'
modelname='MHP'

# The size of each hidden layer except for the last layer
# Use '_' as a delimiter.
dim_mlps='128_64'

# The size of the last hidden layer
dims_hidden_prob=32

dropout=0.0

# Train the model three times with different initialization.
total=3
startFrom=1
for ((i=${startFrom}; i<${startFrom}+$total; i++)); do

    echo starting $i of $total ...

python finetune_classifier.py --gpu ${gpu} --epochs 3 \
    --task_name ${taskname} \
    --surfix ${surfix} \
    --batch_size 32 \
    --seed ${i} \
    --dim_mlps ${dim_mlps} \
    --modelname ${modelname} \
    --num_hidden_prob ${num_hidden_prob} \
    --dims_hidden_prob ${dims_hidden_prob} \
    --dropout ${dropout} \
    --output_dir model_${taskname}_${modelname}_${dt}_${num_hidden_prob}_${dims_hidden_prob}_run${i}_gpu${gpu}_${dim_mlps}_dropout${dropout}

done

