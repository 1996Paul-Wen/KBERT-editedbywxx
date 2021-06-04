#!/bin/bash
cd ..

# CUDA_VISIBLE_DEVICES='0' 

nohup python3 -u run_kbert_ner_ccks2019.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/ccks2019-datasets/train_dev.tsv \
    --dev_path ./datasets/ccks2019-datasets/dev.tsv \
    --test_path ./datasets/ccks2019-datasets/test.tsv \
    --epochs_num 50 --batch_size 16 --kg_name Medical-plus_nocheck \
    --output_model_path ./models/ner_model_bin/bertgru_nocheck.bin \
    > ./outputs/bertgru_nocheck.log 2>&1 &
