import os
import csv
import glob
import json
import numpy as np
from . import utterance

"""
All DataManager classes should follow the following interface:
    1. Input must be a configuration dictionary
    2. All classes must have an assert function that checks if the input is
        valid for designated corpus type
        1) Configuration dictionary must have the following keys:
            - "audio": directory of the audio files
            - "label": path of the label files
    3. Must have a function that returns a list of following items:
        1) List of utterance IDs
        2) List of features
        3) List of categorical labels
        4) List of dimensional labels
"""


class DataManager():
    def __load_config__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return config_dict
    def __init__(self, *args, **kwargs):
        config_path = kwargs.get("config_path", None)
        self.config_dict=self.__load_config__(config_path)
    def get_utts(self, *args, **kwargs):
        raise NotImplementedError
    def get_wav_paths(self, *args, **kwargs):
        raise NotImplementedError
    def get_feat_paths(self, *args, **kwargs):
        raise NotImplementedError
    def get_labels(self, *args, **kwargs):
        raise NotImplementedError

class MSP_Podcast_DataManager(DataManager):
    def __check_env_validity__(self):
        # version = self.config_dict.get("version", None)
        audio_path = self.config_dict.get("audio_path", None)
        label_path = self.config_dict.get("label_path", None)
        if None in [audio_path, label_path]:
            raise ValueError("Invalid environment configuration")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_num = kwargs.get("sample_num", None)

    def get_utts(self, *args, **kwargs):
        """
        Input: split type
            - If split type is None, return all utterances
        Output: list of utterance IDs included in input split type
        """
        split_type = kwargs.get("split_type", None)    
        if split_type != None:
            split_id = self.config_dict["data_split_type"][split_type]

        label_path = self.config_dict["label_path"]
        utt_list=[]
        with open(label_path, 'r') as f:
            f.readline()
            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                stype = row[-1]
                if split_type != None:
                    if stype == split_id:
                        utt_list.append(utt_id) # retrun utterance IDs given split type
                else:
                    utt_list.append(utt_id) # return all utterances
        utt_list.sort()
        return utt_list
    
    def get_wav_paths(self, *args, **kwargs):
        """
        Input: split type
            - If split type is None, return all utterances
        Output: list of utterance IDs included in input split type
        """
        env_type = kwargs.get("env_type", "clean")  
        split_type = kwargs.get("split_type", None)    
        if split_type != None:
            split_id = self.config_dict["data_split_type"][split_type]

        label_path = self.config_dict["label_path"]
        utt_list=[]
        with open(label_path, 'r') as f:
            f.readline()
            csv_reader = csv.reader(f)
            for row in csv_reader:
                utt_id = row[0]
                stype = row[-1]
                if split_type != None:
                    if stype == split_id:
                        utt_list.append(utt_id) # retrun utterance IDs given split type
                else:
                    utt_list.append(utt_id) # return all utterances
        utt_list.sort()
        return utt_list