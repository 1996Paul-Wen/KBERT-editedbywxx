#!/bin/bash
cd ..

CUDA_VISIBLE_DEVICES='0' 

python3 -u run_kbert_ner.py \
    --pretrained_model_path ./models/google_model.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --train_path ./datasets/msra_ner/train.tsv \
    --dev_path ./datasets/msra_ner/dev.tsv \
    --test_path ./datasets/msra_ner/test.tsv \
    --epochs_num 5 --batch_size 16 --kg_name CnDbpedia \
    --output_model_path ./outputs/kbert_msraner_CnDbpedia.bin
