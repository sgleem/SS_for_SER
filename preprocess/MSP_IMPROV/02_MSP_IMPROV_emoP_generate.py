#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 12:27:34 2022

@author: david
"""

import pandas as pd
import numpy as np


#['angry', 'sad', 'neutral', 'happy']
#["Angry","Sad","Neutral","Happy","Other"]
def get_all(input_lst):
    # print(list(input_lst)[0])
    # print(type(list(input_lst)))
    Emotion_count_dic = {'Angry':0,'Sad':0,'Neutral':0,'Happy':0,'Other':0}
 
    for each_emo in list(input_lst)[0]:
        Emotion_count_dic[each_emo]+=1

    return  np.array(list(Emotion_count_dic.values()))
    

#%%
want = ['EmoP']
data = pd.read_pickle('./output/Evalution_raw.pkl')
data_want = data[want]

#%%

All_Dataframe = data_want.apply(get_all,axis=1)


aa_all = []
aa = np.asarray(All_Dataframe.to_numpy())

for each in aa:
    aa_all.append(each)
    
aa_all = np.asarray(aa_all)    

All_Dataframe_final = pd.DataFrame(aa_all,index=data_want.index,columns=['angry', 'sad', 'neutral', 'happy', 'other'])
All_Dataframe_final.to_csv('./output/Primary_Emotion_class_raw_count.csv')
All_Dataframe_final_sum = All_Dataframe_final.sum(axis=0)
All_Dataframe_final_sum.sort_values(0,ascending=False).plot.bar(title="MSP-IMPROV Primary Emotion",legend=0)

# # 