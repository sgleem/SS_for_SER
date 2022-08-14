#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 16:01:30 2022

@author: david
"""

import pandas as pd
import numpy as np

#%%
def str2vec(str_list):
    
    emo_vector = np.array(list(str_list))
    emo_vector = emo_vector/sum(emo_vector)
    
    eps = 0.05
    vector_smo = emo_vector * (1 - eps) + (1-emo_vector) * eps / (len(emo_vector) - 1)
    
    return vector_smo

#%%

Task = 6 #

Method = 'P'

data = pd.read_csv('./output/Emotion_class_raw_count_voice_only.csv',index_col=(0))

if Task == 6:
    emotion_want = ['angry', 'sad', 'disgust', 'fear', 'neutral', 'happy']

data_want = data[emotion_want].apply(str2vec,axis=1)
# #%%

aa_all = []
aa = np.asarray(data_want.to_numpy())

for each in aa:
    aa_all.append(each)
    
aa_all = np.asarray(aa_all)    

All_Dataframe_final = pd.DataFrame(aa_all,index=data_want.index,columns=emotion_want).fillna(0)

#%% 
All_Dataframe_final_dict = All_Dataframe_final.to_dict('index')

for key in All_Dataframe_final_dict:
    value = np.array(list(All_Dataframe_final_dict[key].values()))
    index = np.where(value==np.max(value))[0]
    
    if len(index)==1:
        All_Dataframe_final_dict[key]['Used']=1
    else:
        All_Dataframe_final_dict[key]['Used']=0
    
All_Dataframe_final_p = pd.DataFrame.from_dict(All_Dataframe_final_dict,'index') 
used_or_not = All_Dataframe_final_p['Used'].to_numpy()
count = len(np.where(used_or_not==0.0)[0])
print(Method,' Aggregation Rule')
print('Data Loss:',100*(count/len(used_or_not)),count)
print('Data Use:',100-100*(count/len(used_or_not)),len(used_or_not)-count)
#%% Set Use Or Not
All_Dataframe_final.index.name = 'File_name'
All_Dataframe_final_p.to_csv('./output/labels_consensus_'+str(Task)+'class_'+Method+'.csv')
#%%
All_Dataframe_final_sum = data[emotion_want][All_Dataframe_final_p['Used']==1]
All_Dataframe_final_sum['Used'] = All_Dataframe_final_p['Used']
All_Dataframe_final_sum = All_Dataframe_final_sum.sum()


ax = All_Dataframe_final_sum.plot.bar(legend=0,title="CREMA-D Secondary 6-class Emotion (Annotation-based) "+Method)
ax.figure.savefig('./Distribution/6-class_emoion_voice_only'+Method+'.png')



