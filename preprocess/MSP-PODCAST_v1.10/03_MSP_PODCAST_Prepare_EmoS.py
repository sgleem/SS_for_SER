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
    # ['angry', 'frustrated', 'annoyed', 'disappointed', 'sad', 'disgust', 'depressed', 'contempt', 'confused', 'concerned', 'fear', 'neutral', 'surprise', 'amused', 'excited', 'happy']
    #Emotion_count_dic = {'Angry':0,'Sad':0,'Neutral':0,'Happy':0,'Other':0}
    Emotion_count_dic = {
        'Angry':0,
        'Frustrated':0,
        'Annoyed':0,
        'Disappointed':0,
        'Sad':0,
        'Disgust':0,
        'Depressed':0,
        'Contempt':0,
        'Confused':0,
        'Concerned':0,
        'Fear':0,
        'Neutral':0,
        'Surprise':0,
        'Amused':0,
        'Excited':0,
        'Happy':0,
        'Other':0
        }
    
    input_lst = list(input_lst)[0].split(';')
    for each_emo in input_lst:
        each_emo_list = each_emo.split(',')
        
        for each_one in each_emo_list:
            if 'Other-' in each_one:
                Emotion_count_dic['Other']+=1
            elif each_one in Emotion_count_dic:
                # print(each_one)
                Emotion_count_dic[each_one]+=1
            elif each_one not in Emotion_count_dic:
                if each_one =='Dissapointed':
                    Emotion_count_dic['Disappointed']+=1
                elif each_one =='Surprised':
                    Emotion_count_dic['Surprise']+=1
                elif each_one =='excited':
                    Emotion_count_dic['Excited']+=1
                elif each_one in ['nuetral','neutral']:
                    Emotion_count_dic['Excited']+=1
                else:
                    # print(each_one)
                    continue

    return  np.array(list(Emotion_count_dic.values()))


#%%
path = './output/seconday_dict_v110.pkl'


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

All_Dataframe_final = pd.DataFrame(aa_all,index=data_want.index,columns=['angry', 'frustrated', 'annoyed', 'disappointed', 'sad', 'disgust', 'depressed', 'contempt', 'confused', 'concerned', 'fear', 'neutral', 'surprise', 'amused', 'excited', 'happy', 'other'])    
All_Dataframe_final.to_csv('./output/Secondary_Emotion_class_raw_count.csv')
All_Dataframe_final_sum = All_Dataframe_final.sum(axis=0)
All_Dataframe_final_sum.sort_values(0,ascending=False).plot.bar(title="MSP-PODCAST-Publish-1.10 Secondary Emotion",legend=0)
    

