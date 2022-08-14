#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:53:09 2022

@author: david
"""

import pandas as pd
import numpy as np

#%%
def count2vec(str_list):
    # print(str_list)

    emo_vector = str_list/sum(str_list)
    
    eps = 0.05
    vector_smo = emo_vector * (1 - eps) + (1-emo_vector) * eps / (len(emo_vector) - 1)
    
    return vector_smo

# 4-class
# ['angry', 'sad', 'neutral', 'happy']

Task = 4 #

for Method in ['M','D']:
# Method = 'M'
    if Method == 'M':
        Thresold =  0.5
    elif Method == 'D':    
        Thresold = 0.05
    data = pd.read_csv('./output/Primary_Emotion_class_raw_count.csv',index_col=(0))
    
    
    if Task == 4:
        emotion_want = ['angry', 'sad', 'neutral', 'happy']
        notation = ['A','S','N','H']
        
    #%%
    data_want = data[emotion_want].apply(count2vec,axis=1).fillna(0)   
    data_want.to_csv('./output/Primary_Emotion_'+str(Task)+'class.csv')
    All_Dataframe_final_np = data_want.to_numpy()
    
    #%%
    All_Dataframe_final_check = np.max(All_Dataframe_final_np,axis=1)
    index_list = np.array(list(data_want.index))[np.where(All_Dataframe_final_check>Thresold)[0]]
    count = len(All_Dataframe_final_check)-len(index_list)
    print(Method,' Aggregation Rule')
    print('Data Loss:',100*(count/len(All_Dataframe_final_check)),count)
    print('Data Use:',100-100*(count/len(All_Dataframe_final_check)),len(All_Dataframe_final_check)-count)
    
    #%% Set Use Or Not
    
    used = np.zeros((len(All_Dataframe_final_check),1))
    used[np.where(All_Dataframe_final_check>Thresold)[0]]=1
    
    data_want["Used"] = used
    data_want.index.name = 'File_name'
    data_want.to_csv('./output/Primary_Emotion_'+str(Task)+'class_'+Method+'.csv')








