#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 19:32:53 2022

@author: david
"""

import pandas as pd
import numpy as np
from collections import Counter


#['angry', 'sad', 'disgust', 'excited', 'fear', 'neutral', 'surprise', 'happy']


# 10-class
# ['frustrated'*, 'angry'*, 'sad'*, 'disgust'*, 'excited'*,'fear'*, 'neutral'*, 'surprise', 'happy'*, 'other']

# 9-class
# ['frustrated', 'angry', 'sad', 'disgust', 'excited', 'fear', 'neutral', 'surprise', 'happy']


# 6-class
# ['frustrated', 'angry', 'sad', 'neutral', 'surprise', 'happy']


# 4-class
# ['angry', 'sad', 'neutral', 'happy']


# def get_all(input_lst):

#     #Emotion_count_dic = {'Frustration':0,'Anger':0,'Sadness':0,'Disgust':0,'Fear':0,'Neutral':0,'Surprise':0,'Excited':0,'Happiness':0,'Other':0}
#     Emotion_count_dic = {'frustrated':0,'angry':0,'sad':0,'disgust':0,'fear':0,'neutral':0,'surprise':0,'excited':0,'happy':0,'other':0}
    
    
#     all_data = np.array(input_lst)

#     return ";".join(all_emotion_list)
        
def get_all(input_lst):

    all_list = []
    all_data = list(input_lst)
    all_list.extend(all_data)
    # print(all_list)
    return ";".join(all_list)
        
                
        
    
    # return np.mean(all_data_want)
    

EmoClass = pd.read_csv('./output/Emotion_class_raw.csv',index_col=0).fillna("")
All_Dataframe = pd.DataFrame(index=EmoClass.index)
Want = list(EmoClass.apply(get_all,axis=0))
All_emotion_list = Want[0].split(";")

All_dic = dict(Counter(All_emotion_list))

hist_data = pd.DataFrame.from_dict(All_dic,'index',columns=(['Count']))

ax = hist_data.sort_values(by='Count',ascending=False).plot.bar(title="IEMOCAP All Emotions",legend=0)



