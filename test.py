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
sys.path.append("/media/kyunster/hdd/Project/SS_for_SER_comparison")
import sg_utils_old

from net import ser, chunk, wav2vec2


def main(args):
    seed = args.seed
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.set_deterministic(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    ###################################################################################################
    """
    lab_type: "categorical" or "dimensional"
    For test set,
        test_wavs: list of raw wavs (not a filepath, sampled with 16kHz)
        test_labs: list of labels (categorical: one-hot or normalized vectors)
        test_utts: list of utterances
        => All the lists must be sorted in the same order
    """
    lab_type = args.label_type
    if args.label_type == "dimensional":
        assert args.output_num == 3
    # DataManager=sg_utils.DataManager("conf.json")
    # test_wav_path = DataManager.get_wav_path("msp-podcast", args.data_type, "test", snr=args.snr)
    # test_utts = DataManager.get_utt_list("msp-podcast", "test")
    # test_labs = DataManager.get_msp_labels(test_utts, lab_type=lab_type)

    DataManager=sg_utils_old.DataManager("env/msp_improv.json")
    # DataManager=sg_utils.DataManager("env/msp-podcast/1.10.json")
    test_utts = DataManager.get_utt_list("test")
    test_labs = DataManager.get_msp_labels(test_utts) #, lab_type=lab_type)

    test_wav_path = [DataManager.env_dict["audio_path"]+"/"+utt_id for utt_id in test_utts]
    test_wavs = sg_utils_old.WavExtractor(test_wav_path).extract()
    ###################################################################################################

    test_set = sg_utils_old.WavSet(test_wavs, test_labs, test_utts, print_dur=True, lab_type=lab_type)


    lm = sg_utils_old.LogManager()
    if args.label_type == "dimensional":
        lm.alloc_stat_type_list(["test_aro", "test_dom", "test_val"])
    elif args.label_type == "categorical":
        lm.alloc_stat_type_list(["test_loss", "test_acc"])

    batch_size=args.batch_size
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=sg_utils_old.collate_fn_padd, shuffle=True)
    
    if args.model_type == "manually_finetuned":
        model_path = args.model_path

        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-robust").to(args.device)
        del wav2vec_model.encoder.layers[12:]
        wav2vec_model.load_state_dict(torch.load(model_path+"/final_wav2vec.pt"))
        
        ser_model = ser.HLD(1024, args.hidden_dim, args.num_layers, args.output_num, lab_type=lab_type).to(args.device)
        ser_model.load_state_dict(torch.load(model_path+"/final_head.pt"))
        
        wav2vec_model.eval()
        ser_model.eval()

    elif args.model_type == "msp17_finetuned":
        model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = wav2vec2.EmotionModel.from_pretrained(model_name)
        if args.model_path != None:
            model.load_state_dict(torch.load(args.model_path+"/final.pt"))
        model.to(args.device)
        model.eval()      
    else:
        raise Exception("Invalid model type")  

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
            
            if args.model_type == "manually_finetuned":
                w2v = wav2vec_model(x, attention_mask=mask).last_hidden_state
                h = sg_utils_old.AverageAll(w2v)
                pred = ser_model(h)
                total_pred.append(pred)
                total_y.append(y)

            if args.model_type == "msp17_finetuned":
                pred = model(x)[1]
                total_pred.append(pred)
                total_y.append(y)

            
        total_pred = torch.cat(total_pred, 0)
        total_y = torch.cat(total_y, 0)
    
    if args.label_type == "dimensional":
        ccc = sg_utils_old.CCC_loss(total_pred, total_y)        
        lm.add_torch_stat("test_aro", ccc[0])
        lm.add_torch_stat("test_dom", ccc[1])
        lm.add_torch_stat("test_val", ccc[2])
    elif args.label_type == "categorical":
        loss = sg_utils_old.CE_category(total_pred, total_y)
        acc = sg_utils_old.calc_acc(total_pred, total_y)
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

    # Data Arguments
    parser.add_argument(
        '--data_type',
        default="clean",
        type=str)
    parser.add_argument(
        '--snr',
        default=None,
        type=str)
    parser.add_argument(
        '--feature_type',
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
        '--model_type',
        default="manually_finetuned",
        type=str)
    parser.add_argument(
        '--output_num',
        default=3,
        type=int)
    parser.add_argument(
        '--model_path',
        default=None,
        type=str)
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