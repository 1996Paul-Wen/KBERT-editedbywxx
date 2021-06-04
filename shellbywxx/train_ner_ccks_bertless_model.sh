#!/bin/bash
cd ..

# CUDA_VISIBLE_DEVICES='0' 

nohup python3 -u run_bertless_ner_ccks2019.py \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/ccks2019-datasets/train_dev.tsv \
    --dev_path ./datasets/ccks2019-datasets/dev.tsv \
    --test_path ./datasets/ccks2019-datasets/test.tsv \
    --epochs_num 50 --batch_size 16 --kg_name none --learning_rate 0.001 \
    --output_model_path ./models/ner_model_bin/nf_onlysoftmax_ccks2019_50epoch_nokg.bin \
    > ./outputs/nf_onlysoftmax_ccks2019_50epoch_nokg.log 2>&1 &
