#!/bin/bash

## Training
python -u train.py \
--device            cuda \
--data_type         clean \
--model_type        hubert \
--conf_path         conf.json \
--seed              0 \
--batch_size        16 \
--hidden_dim        1024 \
--num_layers        2 \
--epochs            10 \
--lr                1e-4 \
--noise_dur         30m \
--model_path        model/hubert/1_10/clean_0

## Evaluation
python -u test.py \
--device            cuda \
--data_type         clean \
--model_type        hubert \
--train_type        manually_finetuned \
--conf_path         conf.json \
--seed              0 \
--batch_size        1 \
--hidden_dim        1024 \
--num_layers        2 \
--model_path        model/hubert/1_10/clean_0
