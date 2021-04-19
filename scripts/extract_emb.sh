set -e -x

# prefix is either "train" or "valid".
prefix=$1
modelpath=$2
outdir=$3

cp data/test.bert.csv data/test.bert.csv.bak
cp data/${prefix}.bert.csv data/test.bert.csv 

sh test.sh ${modelpath} ${outdir} 
cp data/test.bert.csv.bak data/test.bert.csv
