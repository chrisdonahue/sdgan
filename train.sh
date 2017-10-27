TRAIN_DIR=./train

python sdgan.py train ${TRAIN_DIR} \
	--data_dir ./data/msceleb12k \
	--data_set msceleb12k \
	--data_id_name_tsv_fp ./data/msceleb12k/train_names.tsv 
