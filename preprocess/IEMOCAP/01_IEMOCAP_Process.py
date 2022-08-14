#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 19:32:53 2022

@author: david
"""

import pandas as pd
import numpy as np

#%%

file_path = 'D:/David/SER_Hackathon/code/IEMOCAP/Emotion.xlsx'

#%%
def get_all(input_lst):
    Emotion_name_list = ['Neutral','Fear','Happiness','Disgust','Sadness','Frustration','Anger','Excited','Surprise']    
    all_data = np.array(input_lst)
    all_data_want = all_data[np.where(all_data!="")]
    
    
    all_emotion_list = []
    
    for each_emo in all_data_want:
        emotion_answers_list = each_emo.split(";")
        for each_answer in emotion_answers_list:
            if each_answer in Emotion_name_list:
                all_emotion_list.append(each_answer)
            else:
                if each_answer == "Other":
                    all_emotion_list.append(each_answer)
    return ";".join(all_emotion_list)

    
EmoClass = pd.read_excel(file_path,index_col=0).fillna("")
All_Dataframe = pd.DataFrame(index=EmoClass.index)
All_Dataframe['EmoClass'] = EmoClass.apply(get_all,axis=1)


All_Dataframe = All_Dataframe.set_index(All_Dataframe.index+'.wav')
All_Dataframe.to_csv('./output/Emotion_class.csv')
#%%

# def count(input_str):
#     Emotion_count_dic = {'Frustration':0,'Anger':0,'Sadness':0,'Disgust':0,'Fear':0,'Neutral':0,'Surprise':0,'Excited':0,'Happiness':0,'Other':0}
#     # print(list(input_str)[0])
#     for each_emotion in list(input_str)[0].split(";"):
#         Emotion_count_dic[each_emotion]+=1
    
#     return list(Emotion_count_dic.values())



# All_Dataframe_dict = All_Dataframe.apply(count,1).to_numpy()
# All_Dataframe_np = np.vstack(All_Dataframe_dict)
# All_Dataframe_pd = pd.DataFrame(All_Dataframe_np,)



