# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
from tqdm import tqdm

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WavLMModel

# Self-Written Modules
sys.path.append("/media/kyunster/hdd/Project/SS_for_SER")
import sg_utils
import net
# from net import ser, chunk


def main(args):
    sg_utils.set_deterministic(args.seed)
    sg_utils.print_config_description(args.conf_path)

    # Make model directory
    model_path = args.model_path
    os.makedirs(model_path+"/param", exist_ok=True)


    # Initialize dataset
    DataManager=sg_utils.DataManager(args.conf_path)
        #     config_path = args.conf_path, 
        #     sample_num = args.sample_num
        # )
    lab_type = args.label_type
    if args.label_type == "dimensional":
        assert args.output_num == 3
    # total_utt_dict={"train": None, "dev": None}
    # total_dataset={"train": None, "dev": None}
    # total_dataloader={"train": None, "dev": None}
    # for dtype in ["train", "dev"]:
    #     total_utt_dict[dtype] = sg_utils.UtteranceList(
    #         DataManager.get_utts(split_type=dtype),
    #         DataManager.get_wav_paths(split_type=dtype),
    #         label = DataManager.get_labels(split_type=dtype, label_type=args.label_type),
    #         label_type = args.label_type,
    #     )
    #     # total_dataset[dtype] = total_utt_dict[dtype].generate_torch_dataset()
    #     total_dataloader[dtype] = total_utt_dict[dtype].generate_torch_dataloader(
    #         normalizer = sg_utils.MSP_Podcast_Normalizer(method="minmax")
    #         batch_size=args.batch_size,
    #         shuffle=True if dtype == "train" else False
    #     )
    ###################################################################################################
    """
    lab_type: "categorical" or "dimensional"
    For training set,
        train_wavs: list of raw wavs (not a filepath, sampled with 16kHz)
        train_labs: list of labels (categorical: one-hot or normalized vectors)
        train_utts: list of utterances
        => All the lists must be sorted in the same order
    For devlopment set,
        dev_wavs: list of raw wavs (not a filepath, sampled with 16kHz)
        dev_labs: list of labels (categorical: one-hot or normalized vectors)
        dev_utts: list of utterances
        => All the lists must be sorted in the same order
    """
    snum=10000000000000000
    train_feat_path = DataManager.get_wav_path("msp-podcast", args.data_type, "train")[:snum]
    train_utts = DataManager.get_utt_list("msp-podcast", "train")[:snum]
    train_labs = DataManager.get_msp_labels(train_utts, lab_type=lab_type)
    train_wavs = sg_utils.WavExtractor(train_feat_path).extract()

    dev_feat_path = DataManager.get_wav_path("msp-podcast", args.data_type, "dev")[:snum]
    dev_utts = DataManager.get_utt_list("msp-podcast", "dev")[:snum]
    dev_labs = DataManager.get_msp_labels(dev_utts)
    dev_wavs = sg_utils.WavExtractor(dev_feat_path).extract()
    ###################################################################################################

    train_set = sg_utils.WavSet(train_wavs, train_labs, train_utts, print_dur=True, lab_type=lab_type)
    dev_set = sg_utils.WavSet(dev_wavs, dev_labs, dev_utts, print_dur=True, lab_type=lab_type,
        wav_mean = train_set.wav_mean, wav_std = train_set.wav_std)
    train_set.save_norm_stat(model_path+"/train_norm_stat.pkl")
    
    total_dataloader={
        "train": DataLoader(train_set, batch_size=args.batch_size, collate_fn=sg_utils.collate_fn_padd, shuffle=True),
        "dev": DataLoader(dev_set, batch_size=args.batch_size, collate_fn=sg_utils.collate_fn_padd, shuffle=False)
    }

    # Initialize model
    modelWrapper = net.ModelWrapper(args) # Change this to use custom model
    modelWrapper.init_model()
    modelWrapper.init_optimizer()

    # Initialize loss function
    lm = sg_utils.LogManager()
    if args.label_type == "dimensional":
        lm.alloc_stat_type_list(["train_aro", "train_dom", "train_val",
            "dev_aro", "dev_dom", "dev_val"])
    elif args.label_type == "categorical":
        lm.alloc_stat_type_list(["train_loss", "train_acc", "dev_loss", "dev_acc"])

    epochs=args.epochs
    scaler = GradScaler()
    min_epoch = 0
    min_loss = 99999999999
    for epoch in range(epochs):
        print("Epoch:",epoch)
        lm.init_stat()
        modelWrapper.set_train()
        for xy_pair in tqdm(total_dataloader["train"]):
            x = xy_pair[0]
            y = xy_pair[1]
            mask = xy_pair[2]

            x=x.cuda(non_blocking=True).float()
            y=y.cuda(non_blocking=True).float()
            mask=mask.cuda(non_blocking=True).float()

            
            with autocast():
                ## Feed-forward
                # w2v = wav2vec_model(x, attention_mask=mask).last_hidden_state
                # h = sg_utils_old.AverageAll(w2v)
                # pred = ser_model(h)
                pred = modelWrapper.feed_forward(x, attention_mask=mask)
                
                ## Calculate loss
                total_loss = 0.0
                if args.label_type == "dimensional":
                    ccc = sg_utils.CCC_loss(pred, y)
                    loss = 1.0-ccc
                    total_loss += loss[0] + loss[1] + loss[2]
                elif args.label_type == "categorical":
                    loss = sg_utils.CE_category(pred, y)
                    total_loss += loss
                    acc = sg_utils.calc_acc(pred, y)
                    

            ## Backpropagation
            modelWrapper.backprop(total_loss)
            # wav2vec_opt.zero_grad(set_to_none=True)
            # ser_opt.zero_grad(set_to_none=True)
            # scaler.scale(total_loss).backward()
            # scaler.step(wav2vec_opt)
            # scaler.step(ser_opt)
            # scaler.update()

            # Logging
            if args.label_type == "dimensional":
                lm.add_torch_stat("train_aro", ccc[0])
                lm.add_torch_stat("train_dom", ccc[1])
                lm.add_torch_stat("train_val", ccc[2])
            elif args.label_type == "categorical":
                lm.add_torch_stat("train_loss", loss)
                lm.add_torch_stat("train_acc", acc)

        # wav2vec_model.eval()
        # ser_model.eval()
        modelWrapper.set_eval()

        with torch.no_grad():
            total_pred = [] 
            total_y = []
            for xy_pair in tqdm(total_dataloader["dev"]):
                x = xy_pair[0]
                y = xy_pair[1]
                mask = xy_pair[2]

                x=x.cuda(non_blocking=True).float()
                y=y.cuda(non_blocking=True).float()
                mask=mask.cuda(non_blocking=True).float()

                # w2v = wav2vec_model(x, attention_mask=mask).last_hidden_state
                # h = sg_utils_old.AverageAll(w2v)
                # pred = ser_model(h)
                pred = modelWrapper.feed_forward(x, attention_mask=mask, eval=True)
                total_pred.append(pred)
                total_y.append(y)

            total_pred = torch.cat(total_pred, 0)
            total_y = torch.cat(total_y, 0)
        
        if args.label_type == "dimensional":
            ccc = sg_utils.CCC_loss(total_pred, total_y)            
            lm.add_torch_stat("dev_aro", ccc[0])
            lm.add_torch_stat("dev_dom", ccc[1])
            lm.add_torch_stat("dev_val", ccc[2])
        elif args.label_type == "categorical":
            loss = sg_utils.CE_category(total_pred, total_y)
            acc = sg_utils.calc_acc(total_pred, total_y)
            lm.add_torch_stat("dev_loss", loss)
            lm.add_torch_stat("dev_acc", acc)


        lm.print_stat()
        if args.label_type == "dimensional":
            dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat("dev_dom") - lm.get_stat("dev_val")
        elif args.label_type == "categorical":
            dev_loss = lm.get_stat("dev_loss")
        if min_loss > dev_loss:
            min_epoch = epoch
            min_loss = dev_loss

        modelWrapper.save_model(epoch)
        # torch.save(wav2vec_model.state_dict(), os.path.join(model_path, "param", str(epoch)+"_wav2vec.pt"))
        # torch.save(ser_model.state_dict(), os.path.join(model_path, "param", str(epoch)+"_head.pt"))
        
    print("Save",end=" ")
    print(min_epoch, end=" ")
    print("")

    print("Loss",end=" ")
    print(3.0-min_loss, end=" ")
    print("")
    modelWrapper.save_final_model(min_epoch, remove_param=True)
    # os.system("cp "+os.path.join(model_path, "param", str(min_epoch)+"_head.pt") + \
    #     " "+os.path.join(model_path, "final_head.pt"))
    # os.system("cp "+os.path.join(model_path, "param", str(min_epoch)+"_wav2vec.pt") + \
    #     " "+os.path.join(model_path, "final_wav2vec.pt"))

    # os.system("rm -rf "+os.path.join(model_path, "param"))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--sample_num',
        default=None,
        type=int)
    parser.add_argument(
        '--conf_path',
        default="conf.json",
        type=str)

    # Data Arguments
    parser.add_argument(
        '--data_type',
        default="clean",
        type=str)
    parser.add_argument(
        '--model_type',
        default="wav2vec",
        type=str)
    parser.add_argument(
        '--label_type',
        choices=['dimensional', 'categorical'],
        default='dimensional',
        type=str)

    # Chunk Arguments
    parser.add_argument(
        '--chunk_window',
        default=50,
        type=int)
    parser.add_argument(
        '--chunk_num',
        default=11,
        type=int)
    
    # Model Arguments
    parser.add_argument(
        '--model_path',
        default="output",
        type=str)
    parser.add_argument(
        '--output_num',
        default=3,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--epochs',
        default=100,
        type=int)
    parser.add_argument(
        '--lr',
        default=1e-5,
        type=float)
    parser.add_argument(
        '--noise_dur',
        default="30m",
        type=str)
    

    args = parser.parse_args()

    # Call main function
    main(args)