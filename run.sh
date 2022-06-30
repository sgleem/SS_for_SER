#!/bin/bash
# Configuration
mtype=hubert
### Model type
## [wav2vec2-base, wav2vec2-large, wav2vec2-large-robust
## hubert-base, hubert-large, 
## wavlm-base, wavlm-base-plus, wavlm-large,
## data2vec-base, data2vec-large]
### Default model type
## wav2vec2: wav2vec2-large-robust
## hubert: hubert-large, 
## wavlm: wavlm-large,
## data2vec: data2vec-large
corpus_type=podcast_v1.7
seed=0

## Training
python -u train.py \
--device            cuda \
--model_type        $mtype \
--corpus_type       $corpus_type \
--seed              $seed \
--batch_size        16 \
--epochs            20 \
--lr                1e-4 \
--hidden_dim        1024 \
--num_layers        2 \
--output_num        10 \
--model_path        model/${mtype}/${corpus_type}/${seed}.json \
--label_type        categorical || exit 1;

## Evaluation
python -u test.py \
--device            cuda \
--model_type        $mtype \
--corpus_type       $corpus_type \
--seed              $seed \
--batch_size        1 \
--hidden_dim        1024 \
--num_layers        2 \
--output_num        10 \
--model_path        model/${mtype}/${corpus_type}/${seed}.json \
--label_type        categorical || exit 1;
