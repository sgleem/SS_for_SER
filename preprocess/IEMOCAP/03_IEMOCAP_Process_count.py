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


def get_all(input_lst):

    Emotion_count_dic = {'Frustration':0,'Anger':0,'Sadness':0,'Disgust':0,'Fear':0,'Neutral':0,'Surprise':0,'Excited':0,'Happiness':0,'Other':0}
    #['frustrated', 'angry', 'sad', 'disgust', 'fear', 'neutral', 'surprise', 'excited', 'happy', 'other']
    # Emotion_count_dic = {'frustrated':0,'angry':0,'sad':0,'disgust':0,'fear':0,'neutral':0,'surprise':0,'excited':0,'happy':0,'other':0}
    
    for each_emo in list(input_lst)[0].split(";"):
        Emotion_count_dic[each_emo]+=1

    
    return  np.array(list(Emotion_count_dic.values()))

    
    # return np.mean(all_data_want)
    

EmoClass = pd.read_csv('./output/Emotion_class_raw.csv',index_col=0).fillna("")
All_Dataframe = pd.DataFrame(index=EmoClass.index)
All_Dataframe = EmoClass.apply(get_all,axis=1)


aa_all = []
aa = np.asarray(All_Dataframe.to_numpy())

for each in aa:
    aa_all.append(each)
    
aa_all = np.asarray(aa_all)    

All_Dataframe_final = pd.DataFrame(aa_all,index=EmoClass.index,columns=['frustrated', 'angry', 'sad', 'disgust', 'fear', 'neutral', 'surprise', 'excited', 'happy', 'other'])
All_Dataframe_final.to_csv('./output/Emotion_class_raw_count.csv')
