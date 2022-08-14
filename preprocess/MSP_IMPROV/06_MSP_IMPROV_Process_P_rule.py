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
Method = 'P'
#%%

for P_or_S in ['P','S']:
    if P_or_S == 'S':
        Task = 10
        data = pd.read_csv('./output/Secondary_Emotion_class_raw_count.csv',index_col=(0))
        emotion_want = ['depressed', 'frustrated', 'angry', 'sad', 'disgust', 'excited', 'fear',
                           'neutral', 'surprise', 'happy']             
    elif P_or_S == 'P':
        Task = 4 #
        data = pd.read_csv('./output/Primary_Emotion_class_raw_count.csv',index_col=(0))
        
        emotion_want = ['angry', 'sad', 'neutral', 'happy']
    
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
    All_Dataframe_final_p.index.name = 'File_name'
    All_Dataframe_final_p.to_csv('./output/labels_consensus_Emo'+P_or_S+'_'+str(Task)+'class_'+Method+'.csv')
    #%%
    All_Dataframe_final_sum = data[emotion_want][All_Dataframe_final_p['Used']==1]
    All_Dataframe_final_sum['Used'] = All_Dataframe_final_p['Used']
    All_Dataframe_final_sum = All_Dataframe_final_sum.sum()
    
    if P_or_S == 'S':
        ax = All_Dataframe_final_sum.plot.bar(legend=0,title="MSP-IMPROV Secondary "+str(Task)+"-class Emotion (Annotation-based) "+Method)
        ax.figure.savefig('./Distribution/'+str(Task)+'-class_Secondary_emoion_'+Method+'.png')
    if P_or_S == 'P':
        ax = All_Dataframe_final_sum.plot.bar(legend=0,title="MSP-IMPROV Primary "+str(Task)+"-class Emotion (Annotation-based) "+Method)
        ax.figure.savefig('./Distribution/'+str(Task)+'-class_Primary_emoion_'+Method+'.png')
    
#%%
    if P_or_S == 'S':
        ax = All_Dataframe_final_sum.plot.bar(legend=0,title="MSP-IMPROV Secondary 4-class Emotion (Annotation-based) "+Method)
        ax.figure.savefig('./Distribution/4-class_Secondary_emoion_'+Method+'.png')
    if P_or_S == 'P':
        ax = All_Dataframe_final_sum.plot.bar(legend=0,title="MSP-IMPROV Primary 4-class Emotion (Annotation-based) "+Method)
        ax.figure.savefig('./Distribution/4-class_Primary_emoion_'+Method+'.png')