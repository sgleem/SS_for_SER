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


for P_or_S in ['P','S']:
    for Method in ['M','D']: 
        if Method == 'M':
            Thresold =  0.5
        elif Method == 'D':    
            Thresold = 0.05
        if P_or_S == 'S':
            data = pd.read_csv('./output/Secondary_Emotion_class_raw_count.csv',index_col=(0))
            Task = 16 #
            emotion_want = ['angry', 'frustrated', 'annoyed', 'disappointed', 'sad', 'disgust', 'depressed', 'contempt', 'confused', 'concerned', 'fear', 'neutral', 'surprise', 'amused', 'excited', 'happy']
        
        elif P_or_S == 'P':
            data = pd.read_csv('./output/Primary_Emotion_class_raw_count.csv',index_col=(0))
            Task = 8 #
            emotion_want = ['angry', 'sad', 'disgust', 'contempt', 'fear', 'neutral', 'surprise', 'happy']

            
        data_want = data[emotion_want].apply(str2vec,axis=1)
        # #%%
        
        aa_all = []
        aa = np.asarray(data_want.to_numpy())
        
        for each in aa:
            aa_all.append(each)
            
        aa_all = np.asarray(aa_all)    
        
        All_Dataframe_final = pd.DataFrame(aa_all,index=data_want.index,columns=emotion_want).fillna(0)
        All_Dataframe_final.to_csv('./output/labels_consensus_Emo'+P_or_S+'_'+str(Task)+'class_original.csv')
        All_Dataframe_final_np = All_Dataframe_final.to_numpy()
        
        #%%
        All_Dataframe_final_check = np.max(All_Dataframe_final_np,axis=1)
        index_list = np.array(list(data_want.index))[np.where(All_Dataframe_final_check>Thresold)[0]]
        count = len(All_Dataframe_final_check)-len(index_list)
        # 
        print(Method,' Aggregation Rule')
        print('Data Loss:',100*(count/len(All_Dataframe_final_check)),count)
        print('Data Use:',100-100*(count/len(All_Dataframe_final_check)),len(All_Dataframe_final_check)-count)
        
        #%% Set Use Or Not
        
        used = np.zeros((len(All_Dataframe_final_check),1))
        used[np.where(All_Dataframe_final_check>Thresold)[0]]=1
        
        All_Dataframe_final["Used"] = used
        All_Dataframe_final.index.name = 'File_name'
        All_Dataframe_final.to_csv('./output/labels_consensus_Emo'+P_or_S+'_'+str(Task)+'class_'+Method+'.csv')
        #%%
        
        All_Dataframe_final_sum = data[emotion_want][All_Dataframe_final['Used']==1]
        All_Dataframe_final_sum['Used'] = All_Dataframe_final['Used']
        All_Dataframe_final_sum = All_Dataframe_final_sum.sum()
        
        
        if P_or_S == 'S':
            ax = All_Dataframe_final_sum.plot.bar(legend=0,title="MSP-PODCAST v1.10 Secondary "+str(Task)+"-class Emotion (Annotation-based) "+Method)
            ax.figure.savefig('./Distribution/'+str(Task)+'-class_Secondary_emoion_'+Method+'.png')
        if P_or_S == 'P':
            ax = All_Dataframe_final_sum.plot.bar(legend=0,title="MSP-PODCAST v1.10 Primary "+str(Task)+"-class Emotion (Annotation-based) "+Method)
            ax.figure.savefig('./Distribution/'+str(Task)+'-class_Primary_emoion_'+Method+'.png')
            
            
            
    
    