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
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# Self-Written Modules
sys.path.append(os.getcwd())
import sg_utils
import net

def main(args):
    sg_utils.set_deterministic(args.seed)
    config_dict = sg_utils.load_env(args.conf_path)
    assert config_dict.get("config_root", None) != None, "No config_root in config/conf.json"
    assert config_dict.get(args.corpus_type, None) != None, "Change config/conf.json"
    config_path = os.path.join(config_dict["config_root"], config_dict[args.corpus_type])
    sg_utils.print_config_description(config_path)
    
    DataManager=sg_utils.DataManager(config_path)
    lab_type = args.label_type
    print(lab_type)
    if args.label_type == "dimensional":
        assert args.output_num == 3
    elif args.label_type == "categorical":
        emo_num = DataManager.get_categorical_emo_num()
        assert args.output_num == emo_num
        
    test_wav_path = DataManager.get_wav_path(split_type="test")
    test_utts = DataManager.get_utt_list("test")
    test_wav_path.sort()
    test_utts.sort()
    test_labs = DataManager.get_msp_labels(test_utts, lab_type=lab_type)
    test_wavs = sg_utils.WavExtractor(test_wav_path).extract()
    ###################################################################################################
    with open(args.model_path+"/train_norm_stat.pkl", 'rb') as f:
        wav_mean, wav_std = pk.load(f)
    test_set = sg_utils.WavSet(test_wavs, test_labs, test_utts, print_dur=True, lab_type=lab_type,
        wav_mean = wav_mean, wav_std = wav_std)


    lm = sg_utils.LogManager()
    if args.label_type == "dimensional":
        lm.alloc_stat_type_list(["test_aro", "test_dom", "test_val"])
    elif args.label_type == "categorical":
        lm.alloc_stat_type_list(["test_loss", "test_acc"])

    batch_size=args.batch_size
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=sg_utils.collate_fn_padd, shuffle=False)
    
    # if args.train_type == "manually_finetuned":
    model_path = args.model_path

    modelWrapper = net.ModelWrapper(args) # Change this to use custom model
    modelWrapper.init_model()
    modelWrapper.load_model(model_path)
    modelWrapper.set_eval()
 

    with torch.no_grad():
        total_pred = [] 
        total_y = []
        for xy_pair in tqdm(test_loader):
            x = xy_pair[0]
            y = xy_pair[1]
            mask = xy_pair[2]
            
            x=x.cuda(non_blocking=True).float()
            y=y.cuda(non_blocking=True).float()
            mask=mask.cuda(non_blocking=True).float()
            
            pred = modelWrapper.feed_forward(x, attention_mask=mask, eval=True)
            total_pred.append(pred)
            total_y.append(y)
            
        total_pred = torch.cat(total_pred, 0)
        total_y = torch.cat(total_y, 0)
    
    if args.label_type == "dimensional":
        ccc = sg_utils.CCC_loss(total_pred, total_y)        
        lm.add_torch_stat("test_aro", ccc[0])
        lm.add_torch_stat("test_dom", ccc[1])
        lm.add_torch_stat("test_val", ccc[2])
    elif args.label_type == "categorical":
        loss = sg_utils.CE_category(total_pred, total_y)
        acc = sg_utils.calc_acc(total_pred, total_y)
        lm.add_torch_stat("test_loss", loss)
        lm.add_torch_stat("test_acc", acc)
    lm.print_stat()

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
        '--conf_path',
        default="config/conf.json",
        type=str)

    # Data Arguments
    parser.add_argument(
        '--data_type',
        default="clean",
        type=str)
    parser.add_argument(
        '--corpus_type',
        default="podcast_v1.7",
        type=str)
    parser.add_argument(
        '--snr',
        default=None,
        type=str)
    parser.add_argument(
        '--model_type',
        default="wav2vec2",
        type=str)
    parser.add_argument(
        '--label_type',
        choices=['dimensional', 'categorical'],
        default='categorical',
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
        default=None,
        type=str)
    parser.add_argument(
        '--output_num',
        default=4,
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