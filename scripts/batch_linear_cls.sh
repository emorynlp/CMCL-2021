emb_path_prefix=$1
for i in 32 64 128; 
do
	suffix=${i}'.'
	nohup python linear_cls.py ${suffix} ${emb_path_prefix} &
done
