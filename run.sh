#!/bin/bash

## Training
python -u train.py \
--device            cuda \
--data_type         clean \
--model_type        wav2vec2 \
--conf_path         conf.json \
--seed              0 \
--batch_size        16 \
--hidden_dim        1024 \
--num_layers        2 \
--epochs            10 \
--lr                1e-4 \
--noise_dur         30m \
--model_path        model/wav2vec2/1_8/clean_0

# ## Evaluation
# python -u test.py \
# --device            cuda \
# --data_type         clean \
# --model_type        wavlm \
# --train_type        manually_finetuned \
# --conf_path         conf.json \
# --seed              0 \
# --batch_size        1 \
# --hidden_dim        1024 \
# --num_layers        2 \
# --model_path        model/wavlm/1_10/clean_0

python -u retrain.py \
--device            cuda \
--data_type         manual_noisy_speech \
--feature_type      wav2vec \
--seed              0 \
--batch_size        16 \
--hidden_dim        1024 \
--num_layers        2 \
--epochs            10 \
--lr                1e-4 \
--noise_dur         30m \
--model_path        model/wav2vec_finetune/10db_random_0