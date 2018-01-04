TRAIN_DIR=./train

#rm -rf ${TRAIN_DIR}
mkdir ${TRAIN_DIR}
git rev-parse HEAD > ${TRAIN_DIR}/git_sha.txt
cp train_dcgan.sh ${TRAIN_DIR}

python sdgan.py train ${TRAIN_DIR} \
	--data_dir ./data/msceleb12k \
	--data_set msceleb12k \
	--data_id_name_tsv_fp ./data/msceleb12k/train_names.tsv 
