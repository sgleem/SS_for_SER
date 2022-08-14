#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 11:06:52 2022

@author: david
"""

import os
import pandas as pd
import numpy as np


path = './Evalution.txt'


Convert_dic = {
    5:1,
    4:2,
    3:3,
    2:4,
    1:5
    }

All_dic = {}
with open(path) as file:
    data = file.readlines()
    for indx,each_line in enumerate(data):
        if each_line =="\n": # Initial
            each_sent_emo = {
                'EmoP':[],
                'EmoS':[],
                'EmoAct':[],
                'EmoVal':[],
                'EmoDom':[],
                'EmoNat':[]
                }
            continue
        else:
            if ".wav" in each_line:
                name = each_line.split(";")[0].replace("UTD-","")
            else:
                each_rater_answers = each_line.replace(" ","").split(";")
                each_primary_emotion = each_rater_answers[1]
                
                for find_a_index,item in enumerate(each_rater_answers):
                    if "A:" in item:
                        record_A_index = find_a_index
                        
                each_secondary_emotion = "-".join(each_rater_answers[2:record_A_index])
                emo_act = float(each_rater_answers[record_A_index].replace("A:",""))
                emo_val = float(each_rater_answers[record_A_index+1].replace("V:",""))
                emo_dom = float(each_rater_answers[record_A_index+2].replace("D:",""))
                emo_nat = float(each_rater_answers[record_A_index+3].replace("N:",""))
                
                
                each_sent_emo['EmoP'].append(each_primary_emotion)
                
                if not (each_primary_emotion in each_secondary_emotion):
                    if 'Other' not in each_primary_emotion: 
                        # print("-"*10)
                        # print(each_primary_emotion)
                        # print(each_secondary_emotion)
                        # print("&"*10)
                        each_secondary_emotion = each_primary_emotion + ',' + each_secondary_emotion
                        # print(each_secondary_emotion)
                        # print("$"*10)
                
                each_sent_emo['EmoS'].append(each_secondary_emotion)
                if not np.isnan(emo_act):
                    each_sent_emo['EmoAct'].append(Convert_dic[int(emo_act)])
                if not np.isnan(emo_val):
                    each_sent_emo['EmoVal'].append(int(emo_val))
                if not np.isnan(emo_dom):
                    each_sent_emo['EmoDom'].append(int(emo_dom))
                if not np.isnan(emo_nat):
                    each_sent_emo['EmoNat'].append(int(emo_nat))
                
        All_dic[name]=each_sent_emo
        


All_df = pd.DataFrame.from_dict(All_dic,'index')    
All_df.to_csv('./output/Evalution_raw.csv')
All_df.to_pickle('./output/Evalution_raw.pkl')





