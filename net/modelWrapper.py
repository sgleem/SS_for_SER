import os
import sys
from . import wav2vec2
from . import ser
from transformers import Wav2Vec2Model, WavLMModel, HubertModel, Data2VecAudioModel
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
# import fairseq
sys.path.append("/media/kyunster/hdd/Project/SS_for_SER")
import sg_utils

class ModelWrapper():
    def __init__(self, args, **kwargs):
        self.args = args

        self.device = args.device
        self.model_type = args.model_type

        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.output_num = args.output_num
        self.lab_type = args.label_type

        self.lr = args.lr

        self.model_path = args.model_path
        return


    def init_model(self):
        """
        Define model and load pretrained weights
        """
        assert self.model_type in ["data2vec", "wav2vec2", "hubert", "wavlm"], \
            print("Wrong model type")
        # If base model, set it to False
        if self.model_type == "wav2vec2":
            print("Loading wav2vec2-large-robust model")
            self.wav2vec_model= Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-robust")
            del self.wav2vec_model.encoder.layers[12:]
            self.wav2vec_model.freeze_feature_encoder()
            is_large = True
        
        elif self.model_type == "data2vec":
            print("Loading data2vec-audio-base-960h model")
            self.wav2vec_model= Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")
            # del self.wav2vec_model.encoder.layers[12:]
            self.wav2vec_model.freeze_feature_encoder()
            is_large = True

        # elif self.model_type == "wav2vec1":
        #     print("Loading wav2vec1.0-large model")
        #     os.makedirs("pretrained", exist_ok=True)
        #     os.system("wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt -O pretrained/wav2vec-large.pt")
        #     model, _, _= fairseq.checkpoint_utils.load_model_ensemble_and_task(["pretrained/wav2vec-large.pt"])
        #     self.wav2vec_model = model[0]
        #     # del self.wav2vec_model.encoder.layers[12:]
        #     is_large = True 

        elif self.model_type == "hubert":
            print("Loading HuBert-large model")
            self.wav2vec_model= HubertModel.from_pretrained("facebook/hubert-large-ll60k")
            self.wav2vec_model.feature_extractor._freeze_parameters()
            is_large = True 

        elif self.model_type == "wavlm":
            print("Loading WavLM-large model")
            self.wav2vec_model= WavLMModel.from_pretrained("microsoft/wavlm-large")
            self.wav2vec_model.freeze_feature_encoder()
            is_large = True
        if self.model_type == "wav2vec1":
            idim = 512
        elif is_large:
            idim = 1024
        else:
            idim = 768
        self.ser_model = ser.HLD(
            idim,
            self.hidden_dim, 
            self.num_layers, 
            self.output_num, 
            p=0.5, 
            lab_type=self.lab_type)

        self.wav2vec_model.to(self.device)
        
        self.ser_model.to(self.device)

    def init_optimizer(self):
        """
        Define optimizer for pre-trained model
        """
        assert self.wav2vec_model is not None and self.ser_model is not None, \
            print("Model is not initialized")
        
        self.wav2vec_opt = optim.Adam(self.wav2vec_model.parameters(), lr=self.lr)
        self.ser_opt = optim.Adam(self.ser_model.parameters(), lr=self.lr)
        self.scaler = GradScaler()
    
    def feed_forward(self, x, eval=False, **kwargs):
        """
        Feed forward the model
        """
        def __inference__(self, x, **kwargs):
            mask = kwargs.get("attention_mask", None)
            if self.model_type == "wav2vec1":
                z = self.wav2vec_model.feature_extractor(x)
                w2v = self.wav2vec_model.feature_aggregator(z)
            else:
                w2v = self.wav2vec_model(x, attention_mask=mask).last_hidden_state
            h = sg_utils.AverageAll(w2v)
            pred = self.ser_model(h)
            return pred
        
        if eval:
            with torch.no_grad():
                return __inference__(self, x, **kwargs)
        else:
            return __inference__(self, x, **kwargs)
    
    def backprop(self, total_loss):
        """
        Update the model given loss
        """
        self.wav2vec_opt.zero_grad(set_to_none=True)
        self.ser_opt.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.wav2vec_opt)
        self.scaler.step(self.ser_opt)
        self.scaler.update()

    def save_model(self, epoch):
        """
        Save the model for each epoch
        """

        torch.save(self.wav2vec_model.state_dict(), \
            os.path.join(self.model_path, "param", str(epoch)+"_wav2vec.pt"))
        torch.save(self.ser_model.state_dict(), \
            os.path.join(self.model_path, "param", str(epoch)+"_head.pt"))
    
    def save_final_model(self, min_epoch, remove_param=False):
        """
        Copy the given epoch model to the final model
            if remove_param is True, remove the param folder
        """
        os.system("cp "+os.path.join(self.model_path, "param", str(min_epoch)+"_head.pt") + \
        " "+os.path.join(self.model_path, "final_head.pt"))
        os.system("cp "+os.path.join(self.model_path, "param", str(min_epoch)+"_wav2vec.pt") + \
            " "+os.path.join(self.model_path, "final_wav2vec.pt"))

        if remove_param:
            os.system("rm -rf "+os.path.join(self.model_path, "param"))

    def set_eval(self):
        """
        Set the model to eval mode
        """
        self.wav2vec_model.eval()
        self.ser_model.eval()
    def set_train(self):
        """
        Set the model to train mode
        """
        self.wav2vec_model.train()
        self.ser_model.train()

    def load_model(self, model_path):
        self.wav2vec_model.load_state_dict(torch.load(model_path+"/final_wav2vec.pt"))
        self.ser_model.load_state_dict(torch.load(model_path+"/final_head.pt"))