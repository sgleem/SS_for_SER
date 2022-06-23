#!/bin/bash

for seed in 0; do
    ## Training
    python -u train.py \
    --device            cuda \
    --data_type         clean \
    --model_type        wav2vec2 \
    --conf_path         conf.json \
    --seed              ${seed} \
    --batch_size        16 \
    --hidden_dim        1024 \
    --num_layers        2 \
    --epochs            10 \
    --lr                1e-4 \
    --model_path        model/wav2vec2/1_8/clean_${seed}

    # ## Evaluation
    python -u test.py \
    --device            cuda \
    --data_type         clean \
    --model_type        wav2vec2 \
    --train_type        manually_finetuned \
    --conf_path         conf.json \
    --seed              ${seed} \
    --batch_size        1 \
    --hidden_dim        1024 \
    --num_layers        2 \
    --model_path        model/wav2vec2/1_8/clean_${seed}

    # for snr in 10db 5db 0db; do
    #     echo $snr
    #     python -u test.py \
    #     --device            cuda \
    #     --data_type         noisy \
    #     --snr               $snr \
    #     --model_type        wav2vec2 \
    #     --train_type        manually_finetuned \
    #     --conf_path         conf.json \
    #     --seed              ${seed} \
    #     --batch_size        1 \
    #     --hidden_dim        1024 \
    #     --num_layers        2 \
    #     --model_path        model/wav2vec2/1_8/clean_${seed}
    # done

    # # augmented noise
    # python -u retrain.py \
    # --device            cuda \
    # --data_type         manual_noisy_speech \
    # --snr               10db \
    # --model_type        wav2vec2 \
    # --seed              ${seed} \
    # --batch_size        16 \
    # --hidden_dim        1024 \
    # --num_layers        2 \
    # --epochs            10 \
    # --lr                1e-4 \
    # --noise_dur         30m \
    # --original_model_path model/wav2vec2/1_8/clean_${seed} \
    # --model_path        model/wav2vec2_finetune/1_8/10db_noisysample_${seed}

    # # ## Evaluation
    # echo clean
    # python -u test.py \
    # --device            cuda \
    # --data_type         clean \
    # --model_type        wav2vec2 \
    # --train_type        manually_finetuned \
    # --conf_path         conf.json \
    # --seed              ${seed} \
    # --batch_size        1 \
    # --hidden_dim        1024 \
    # --num_layers        2 \
    # --model_path        model/wav2vec2_finetune/1_8/10db_noisysample_${seed}

    # for snr in 10db 5db 0db; do
    #     echo $snr
    #     python -u test.py \
    #     --device            cuda \
    #     --data_type         noisy \
    #     --snr               $snr \
    #     --model_type        wav2vec2 \
    #     --train_type        manually_finetuned \
    #     --conf_path         conf.json \
    #     --seed              ${seed} \
    #     --batch_size        1 \
    #     --hidden_dim        1024 \
    #     --num_layers        2 \
    #     --model_path        model/wav2vec2_finetune/1_8/10db_noisysample_${seed}
    # done
    # # real noise
    # python -u retrain.py \
    # --device            cuda \
    # --data_type         noisy \
    # --snr               10db \
    # --model_type        wav2vec2 \
    # --seed              ${seed} \
    # --batch_size        16 \
    # --hidden_dim        1024 \
    # --num_layers        2 \
    # --epochs            10 \
    # --lr                1e-4 \
    # --noise_dur         30m \
    # --original_model_path model/wav2vec2/1_8/clean_${seed} \
    # --model_path        model/wav2vec2_finetune/1_8/10db_real_noisy_${seed}

    # # ## Evaluation
    # echo clean
    # python -u test.py \
    # --device            cuda \
    # --data_type         clean \
    # --model_type        wav2vec2 \
    # --train_type        manually_finetuned \
    # --conf_path         conf.json \
    # --seed              ${seed} \
    # --batch_size        1 \
    # --hidden_dim        1024 \
    # --num_layers        2 \
    # --model_path        model/wav2vec2_finetune/1_8/10db_real_noisy_${seed}

    # for snr in 10db 5db 0db; do
    #     echo $snr
    #     python -u test.py \
    #     --device            cuda \
    #     --data_type         noisy \
    #     --snr               $snr \
    #     --model_type        wav2vec2 \
    #     --train_type        manually_finetuned \
    #     --conf_path         conf.json \
    #     --seed              ${seed} \
    #     --batch_size        1 \
    #     --hidden_dim        1024 \
    #     --num_layers        2 \
    #     --model_path        model/wav2vec2_finetune/1_8/10db_real_noisy_${seed}
    # done
done