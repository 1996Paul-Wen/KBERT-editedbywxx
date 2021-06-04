#!/bin/bash
cd ..

# CUDA_VISIBLE_DEVICES='0' 

nohup python3 -u run_kbertless_ner_predict.py \
	    --trained_model_path ./models/ner_model_bin/f_grucrf_ccks2019_50epoch_nokg.bin \
	        --config_path ./models/google_config.json \
		    --vocab_path ./models/google_vocab.txt \
		        --train_path ./datasets/ccks2019-datasets/train_dev.tsv \
			    --test_path ./datasets/ccks2019-datasets/test.tsv \
			        --batch_size 16 --kg_name none \
				    > ./outputs/s_grucrf_ccks2019_50epoch_nokg.log 2>&1 &
