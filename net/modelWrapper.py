import os
import sys
from . import wav2vec2
from . import ser
from transformers import Wav2Vec2Model, WavLMModel, HubertModel
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
sys.path.append(os.getcwd())
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
        assert self.model_type in [
            "wav2vec2", "hubert", "wavlm", "data2vec",
            "wav2vec2-base", "wav2vec2-large", "wav2vec2-large-robust",
            "hubert-base", "hubert-large",
            "wavlm-base", "wavlm-base-plus", "wavlm-large",
            "data2vec-base", "data2vec-large"], \
            print("Wrong model type")
        # If base model, set is_large to False
        default_models={
            "wav2vec2": "wav2vec2-large-robust",
            "hubert": "hubert-large",
            "wavlm": "wavlm-large",
            "data2vec": "data2vec-large",
        }
        real_model_name = default_models.get(self.model_type, self.model_type)

        if real_model_name == "wav2vec2":
            self.wav2vec_model= Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-robust")
            del self.wav2vec_model.encoder.layers[12:]
            is_large = True 

        elif real_model_name == "hubert":
            self.wav2vec_model= HubertModel.from_pretrained("facebook/hubert-large-ll60k")
            is_large = True 

        elif real_model_name == "wavlm":
            # self.wav2vec_model= WavLMModel.from_pretrained("microsoft/wavlm-large")
            self.wav2vec_model= WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
            is_large = False

        self.ser_model = ser.HLD(
            1024 if is_large else 768, 
            self.hidden_dim, 
            self.num_layers, 
            self.output_num, 
            p=0.5, 
            lab_type=self.lab_type)

        self.wav2vec_model.to(self.device)
        self.wav2vec_model.freeze_feature_encoder()
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