TRAIN_DIR=./train

export CUDA_VISIBLE_DEVICES="-1"
python sdgan.py preview ${TRAIN_DIR} \
	--data_set msceleb12k \
	--preview_nids 8 \
	--preview_nobs 6
