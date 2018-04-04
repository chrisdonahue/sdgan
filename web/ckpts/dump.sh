rm -rf dumped
python dump_checkpoint_vars.py \
	--model tensorflow \
	--checkpoint_file ${1} \
	--remove_variables_regex '^((beta.*)|(D.*)|(.*Adam.*)|(loader.*))$' \
	--output_dir dumped
