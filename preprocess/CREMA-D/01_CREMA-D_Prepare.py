#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 14:00:26 2022

@author: david
"""

import pandas as pd
import numpy as np
#%%

file_path = 'D:/David/SER_Hackathon/code/CREMA-D/finishedResponses.csv'


#%%
"""
"A" - count of Anger Responses
"D" - count of Disgust Responses
"F" - count of Fear Responses
"H" - count of Happy Responses
"N" - count of Neutral Responses
"S" - count of Sad Responses
"""

"""
localid: annotator
ans: emotion character with level separated by an underscore
"""

Emotion_dic = {
    "A" : 'Anger',
    "D" : 'Disgust',
    "F" : 'Fear',
    "H" : 'Happy',
    "N" : 'Neutral',
    "S" : 'Sad'
    }

#%%
def get_all(input_lst):
    
    Emotion_count_dic = {
        'Anger':0,
        'Sad':0,
        'Disgust':0,
        'Fear':0,
        'Neutral':0,
        'Happy':0
        }
    
    input_lst = list(input_lst)
    for each_emo in input_lst:
        if each_emo in Emotion_count_dic:
            Emotion_count_dic[each_emo]+=1
        else:
            if each_emo!=None:
                print(each_emo)
            continue

    return  np.array(list(Emotion_count_dic.values()))
#%%




Want_cols = ['clipName','localid','respEmo']

data = pd.read_csv(file_path)

data_audio = data[data["queryType"]==1][Want_cols]
# data_audio = data[Want_cols]
#%%
data_audio_dict = data_audio.to_dict('index')

All_dic = {}
for key in data_audio_dict:
    file_name = data_audio_dict[key]['clipName']+".wav"
    if file_name not in All_dic:
        All_dic[file_name] = []

    rater = data_audio_dict[key]['localid']
    emotion = Emotion_dic[data_audio_dict[key]['respEmo']]
    if len(All_dic[file_name])==0:
        All_dic[file_name] = [emotion]
    else:
        All_dic[file_name].append(emotion)

data_want = pd.DataFrame.from_dict(All_dic,'index')    

#%%
All_Dataframe = data_want.apply(get_all,axis=1) 
    
aa_all = []
aa = np.asarray(All_Dataframe.to_numpy())

for each in aa:
    aa_all.append(each)
    
aa_all = np.asarray(aa_all)      
    

All_Dataframe_final = pd.DataFrame(aa_all,index=data_want.index,columns=['angry', 'sad', 'disgust', 'fear', 'neutral', 'happy'])    
All_Dataframe_final.to_csv('./output/Emotion_class_raw_count_voice_only.csv')
All_Dataframe_final_sum = All_Dataframe_final.sum(axis=0)
All_Dataframe_final_sum.sort_values(0,ascending=False).plot.bar(title="CREMA-D Emotion (Voice Only)",legend=0)
        
    
    
    
    






