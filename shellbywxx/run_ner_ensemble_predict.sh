#!/bin/bash
cd ..

# CUDA_VISIBLE_DEVICES='0' 

nohup python3 -u run_kbert_ner_ensemble.py \
    --trained_model_path /home/wxx/workspace/K-BERT/models/ner_model_bin/bertcrf_nocheck.bin,/home/wxx/workspace/K-BERT/models/ner_model_bin/bertgrucrf_nocheck.bin,/home/wxx/workspace/K-BERT/models/ner_model_bin/bertlstmcrf_nocheck.bin,/home/wxx/workspace/K-BERT/models/ner_model_bin/bertsoftmaxcrossdrop_nocheck.bin,/home/wxx/workspace/K-BERT/models/ner_model_bin/bertlstm_nocheck.bin,/home/wxx/workspace/K-BERT/models/ner_model_bin/bertgru_nocheck.bin \
    --config_path ./models/google_config.json \
    --vocab_path ./models/google_vocab.txt \
    --predict_data_path ./datasets/ccks2019-datasets/test.tsv \
    --train_path ./datasets/ccks2019-datasets/train_dev.tsv \
    --batch_size 16 --kg_name Medical-plus_nocheck \
    > ./outputs/ensemble_nocheck_basebgc.log 2>&1 &
