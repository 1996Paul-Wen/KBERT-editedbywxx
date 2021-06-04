#!/bin/bash
cd ..

# CUDA_VISIBLE_DEVICES='0' 

nohup python3 -u run_cls_predict_bywxx.py \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --trained_model_path ./models/cls_model_bin/graduate_senti_512cls.bin \
    --predict_path ./datasets/g_s/5_not_match_senti.tsv \
    --batch_size 16 --kg_name CnDbpedia --labels_num 2 --predict_output_path ./outputs/5_predict_output.txt \
    > ./outputs/predict_senti_5.log 2>&1 &
