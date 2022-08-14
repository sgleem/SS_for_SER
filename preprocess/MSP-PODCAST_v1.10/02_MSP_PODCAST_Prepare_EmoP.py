#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:21:16 2022

@author: david
"""


import pandas as pd
import joblib
import numpy as np 
#%%
def get_all(input_lst):
    # print(list(input_lst)[0])
    # print(type(list(input_lst)))
    # ['angry', 'sad', 'disgust', 'contempt', 'fear', 'neutral', 'surprise', 'happy']
    #Emotion_count_dic = {'Angry':0,'Sad':0,'Neutral':0,'Happy':0,'Other':0}
    Emotion_count_dic = {
        'Angry':0,
        'Sad':0,
        'Disgust':0,
        'Contempt':0,
        'Fear':0,
        'Neutral':0,
        'Surprise':0,
        'Happy':0,
        'Other':0
        }
    
    input_lst = list(input_lst)[0].split(';')
    
    for each_emo in input_lst:
        if 'Other-' in each_emo:
            Emotion_count_dic['Other']+=1
        else:
            Emotion_count_dic[each_emo]+=1

    return  np.array(list(Emotion_count_dic.values()))


#%%
path = './output/primary_dict_v110.pkl'


data_dic = joblib.load(path)


All_dic= {}

for key in data_dic:
    each_data = list(data_dic[key].values())
    All_dic[key] = ";".join(each_data)

data_want = pd.DataFrame.from_dict(All_dic,'index')

#%%
All_Dataframe = data_want.apply(get_all,axis=1)    
    
    
aa_all = []
aa = np.asarray(All_Dataframe.to_numpy())

for each in aa:
    aa_all.append(each)
    
aa_all = np.asarray(aa_all)    

All_Dataframe_final = pd.DataFrame(aa_all,index=data_want.index,columns=['angry', 'sad', 'disgust', 'contempt', 'fear', 'neutral', 'surprise', 'happy', 'other'])    
All_Dataframe_final.to_csv('./output/Primary_Emotion_class_raw_count.csv')
All_Dataframe_final_sum = All_Dataframe_final.sum(axis=0)
All_Dataframe_final_sum.sort_values(0,ascending=False).plot.bar(title="MSP-PODCAST-Publish-1.10 Primary Emotion",legend=0)
    

