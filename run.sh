#!/bin/bash


python -u train.py \
--device            cuda \
--data_type         clean \
--feature_type      wavlm \
--seed              0 \
--batch_size        16 \
--hidden_dim        1024 \
--num_layers        2 \
--epochs            10 \
--lr                1e-4 \
--noise_dur         30m \
--model_path        model/wavlm/1_10/clean_0

# python -u test.py \
# --device            cuda \
# --data_type         clean \
# --feature_type      wav2vec \
# --model_type        manually_finetuned \
# --seed              0 \
# --batch_size        1 \
# --hidden_dim        1024 \
# --num_layers        2 \
# --model_path        model/wav2vec/1_10/clean_0

# python -u test.py \
# --device            cuda \
# --data_type         clean \
# --feature_type      wav2vec \
# --model_type        msp17_finetuned \
# --seed              0 \
# --batch_size        1 \
# --hidden_dim        1024 \
# --num_layers        2 \

# python -u test.py \
# --device            cuda \
# --data_type         clean \
# --feature_type      wav2vec \
# --model_type        msp17_finetuned \
# --seed              0 \
# --batch_size        1 \
# --hidden_dim        1024 \
# --num_layers        2 \
# --model_path        model/wav2vec/1_7/clean_0

# python -u test.py \
# --device            cuda \
# --data_type         noisy \
# --snr               10db \
# --feature_type      wav2vec \
# --model_type        msp17_finetuned \
# --seed              0 \
# --batch_size        1 \
# --hidden_dim        1024 \
# --num_layers        2 \
# --model_path        model/wav2vec_finetune/10db_noisysample_0

# python -u test.py \
# --device            cuda \
# --data_type         noisy \
# --snr               10db \
# --feature_type      wav2vec \
# --model_type        msp17_finetuned \
# --seed              0 \
# --batch_size        1 \
# --hidden_dim        1024 \
# --num_layers        2 \
# --model_path        model/wav2vec_finetune/10db_random_0

# python -u finetune_ser_w2v2.py \
# --device            cuda \
# --data_type         viewmaker_random \
# --feature_type      wav2vec \
# --seed              0 \
# --batch_size        16 \
# --hidden_dim        1024 \
# --num_layers        2 \
# --epochs            10 \
# --lr                1e-4 \
# --noise_dur         30m \
# --model_path        model/wav2vec_finetune/10db_random_0