#!/bin/bash

## Training
python -u train.py \
--device            cuda \
--data_type         clean \
--model_type        wav2vec2 \
--conf_path         conf.json \
--seed              0 \
--batch_size        17 \
--hidden_dim        1024 \
--num_layers        2 \
--epochs            20 \
--lr                1e-4 \
--noise_dur         30m \
--model_path        /home/david/SER_Hackathon/code/model/SS_for_SER/model/USC-IEMOCAP/wav2vec2/partitions_1 \
--label_type        categorical 

## Evaluation
python -u test.py \
--device            cuda \
--data_type         clean \
--model_type        wav2vec2 \
--train_type        manually_finetuned \
--conf_path         conf.json \
--seed              0 \
--batch_size        1 \
--hidden_dim        1024 \
--num_layers        2 \
--model_path        /home/david/SER_Hackathon/code/model/SS_for_SER/model/USC-IEMOCAP/wav2vec2/partitions_1 \
--label_type        categorical 

# python -u test.py \
# --device            cuda \
# --data_type         noisy \
# --snr               10db \
# --model_type        wav2vec2 \
# --train_type        manually_finetuned \
# --conf_path         conf.json \
# --seed              0 \
# --batch_size        1 \
# --hidden_dim        1024 \
# --num_layers        2 \
# --model_path        /home/david/SER_Hackathon/code/model/SS_for_SER/model/wav2vec2

# python -u retrain.py \
# --device            cuda \
# --data_type         manual_noisy_speech \
# --model_type      wav2vec2 \
# --seed              0 \
# --batch_size        16 \
# --hidden_dim        1024 \
# --num_layers        2 \
# --epochs            10 \
# --lr                1e-4 \
# --noise_dur         30m \
# --original_model_path model/wav2vec2/1_8/clean_0 \
# --model_path        model/wav2vec2_finetune/1_8/10db_noisysample_0