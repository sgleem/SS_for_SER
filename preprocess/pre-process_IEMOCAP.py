#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:24:13 2021

@author: lucas
"""

#%%CREMA-D Frame extraction
from distutils import file_util
import numpy as np
import os.path
import os 
from glob import glob
import pandas as pd
#%% Frames Extraction
from scipy.io import savemat
folds = ['Ses01','Ses02','Ses03','Ses04','Ses05']
loc = '/home/podcast/Desktop/MSP-Hackathon/USC-IEMOCAP/Audios/'

waves = os.listdir(loc)

parts = {}

for f in folds:
    temp = []
    # print(sess)
    for wav in waves:
        if wav[0:5] == f:
            temp.append(wav)
    parts[f] = temp


print(parts.keys())

# filename = 'partitions_5.txt'

# train = ['Ses02','Ses03','Ses04']
# test = ['Ses05']
# development = ['Ses01']

# with open(filename, 'w') as f:
#     for val in train:
#         for fname in parts[val]:
#             text = 'Train; '+ fname + '\n'
#             f.write(text)
#     for val in test:
#         for fname in parts[val]:
#             text = 'Test; '+ fname + '\n'
#             f.write(text)
#     for val in development:
#         for fname in parts[val]:
#             text = 'Development; '+ fname + '\n'
#             f.write(text)


# print(filename)
# print(train)
# print(test)
# print(development)

values_emo = ['P']

for val_emo in values_emo: 

    e_file = 'labels_consensus_4class_' + val_emo
    emo_file = e_file + '.csv'

    label_consensus_loc = '/home/podcast/Desktop/MSP-Hackathon/USC-IEMOCAP/USC-IEMOCAP_processing/Attributes/labels_consensus_attritutes.csv'
    categ_labels = '/home/podcast/Desktop/MSP-Hackathon/USC-IEMOCAP/USC-IEMOCAP_processing/labels_processed/' + emo_file
    
    dir_to_save = '/home/podcast/Desktop/MSP-Hackathon/USC-IEMOCAP/IEMOCAP_partitions/'+ e_file

    os.mkdir(dir_to_save)

    att_lbl = pd.read_csv (label_consensus_loc)
    categ_lbl = pd.read_csv (categ_labels)

    # print(categ_lbl)

    numbers = ['1','2','3','4','5']
    for number in numbers:

        partition = 'partitions_' + number + '.txt'

        file_loc = '/home/podcast/Desktop/MSP-Hackathon/USC-IEMOCAP/partitions/' + partition

        file_part = open(file_loc, 'r')
        Lines_part = file_part.readlines()



        IEMOCAP_PARTITIONS = {}
        NAN_list = []

        for line in Lines_part:
            line = line.strip().split(';')
            IEMOCAP_PARTITIONS[line[1][1:]] = line[0]

        # print(IEMOCAP_PARTITIONS)


        value = att_lbl.loc[att_lbl['File_name']=='Ses05M_script03_2_M042.wav']

        print(value['EmoAct'].to_string(index=False))



        CONSENSUS = {}

        fname, emo_a, emo_s, emo_n, emo_h, act, val, dom, s_id, gender, split = [],[],[],[],[],[],[],[],[],[],[]

        for key in IEMOCAP_PARTITIONS.keys():
            value_categ = categ_lbl.loc[categ_lbl['File_name']==key]

            used = value_categ['Used'].to_string(index=False)

            if used == '1':
                value_attr = att_lbl.loc[att_lbl['File_name']==key]
                fname.append(key)
                emo_a.append(value_categ['angry'].to_string(index=False))
                emo_s.append(value_categ['sad'].to_string(index=False))
                emo_n.append(value_categ['neutral'].to_string(index=False))
                emo_h.append(value_categ['happy'].to_string(index=False))
                act.append(value_attr['EmoAct'].to_string(index=False))
                val.append(value_attr['EmoVal'].to_string(index=False))
                dom.append(value_attr['EmoDom'].to_string(index=False))
                s_id.append(key[3:5])
                if key[-8:-7] == 'M':
                    gender.append('Male')
                elif key[-8:-7] == 'F':
                    gender.append('Female')
                else:
                    print(key)
                split.append(IEMOCAP_PARTITIONS[key])


        CONSENSUS['FileName'] = fname
        CONSENSUS['angry'] = emo_a
        CONSENSUS['sad'] = emo_s
        CONSENSUS['neutral'] = emo_n
        CONSENSUS['happy'] = emo_h
        CONSENSUS['EmoAct'] = act
        CONSENSUS['EmoVal'] = val
        CONSENSUS['EmoDom'] = dom
        CONSENSUS['SpkrID'] = s_id
        CONSENSUS['Gender'] = gender
        CONSENSUS['Split_Set'] = split

        print(max(dom), max(val), max(act))
        print(min(dom), min(val), min(act))

        # print(CONSENSUS)


        df = pd.DataFrame(CONSENSUS)

        name = dir_to_save + '/labels_consensus_' + number + '.csv'
        # print(CONSENSUS)
        
        df.to_csv(name,index=False)





#%% 9 class

# #%% Frames Extraction
# from scipy.io import savemat
# folds = ['Ses01','Ses02','Ses03','Ses04','Ses05']
# loc = '/home/podcast/Desktop/MSP-Hackathon/USC-IEMOCAP/Audios/'

# waves = os.listdir(loc)

# parts = {}

# for f in folds:
#     temp = []
#     # print(sess)
#     for wav in waves:
#         if wav[0:5] == f:
#             temp.append(wav)
#     parts[f] = temp


# print(parts.keys())

# # filename = 'partitions_5.txt'

# # train = ['Ses02','Ses03','Ses04']
# # test = ['Ses05']
# # development = ['Ses01']

# # with open(filename, 'w') as f:
# #     for val in train:
# #         for fname in parts[val]:
# #             text = 'Train; '+ fname + '\n'
# #             f.write(text)
# #     for val in test:
# #         for fname in parts[val]:
# #             text = 'Test; '+ fname + '\n'
# #             f.write(text)
# #     for val in development:
# #         for fname in parts[val]:
# #             text = 'Development; '+ fname + '\n'
# #             f.write(text)


# # print(filename)
# # print(train)
# # print(test)
# # print(development)

# values_emo = ['P']

# for val_emo in values_emo: 

#     e_file = 'labels_consensus_9class_' + val_emo
#     emo_file = e_file + '.csv'

#     label_consensus_loc = '/home/podcast/Desktop/MSP-Hackathon/USC-IEMOCAP/USC-IEMOCAP_processing/Attributes/labels_consensus_attritutes.csv'
#     categ_labels = '/home/podcast/Desktop/MSP-Hackathon/USC-IEMOCAP/All_Emotions/' + emo_file
    
#     dir_to_save = '/home/podcast/Desktop/MSP-Hackathon/USC-IEMOCAP/IEMOCAP_partitions/'+ e_file

#     os.mkdir(dir_to_save)

#     att_lbl = pd.read_csv (label_consensus_loc)
#     categ_lbl = pd.read_csv (categ_labels)

#     # print(categ_lbl)

#     numbers = ['1','2','3','4','5']
#     for number in numbers:

#         partition = 'partitions_' + number + '.txt'

#         file_loc = '/home/podcast/Desktop/MSP-Hackathon/USC-IEMOCAP/partitions/' + partition

#         file_part = open(file_loc, 'r')
#         Lines_part = file_part.readlines()



#         IEMOCAP_PARTITIONS = {}
#         NAN_list = []

#         for line in Lines_part:
#             line = line.strip().split(';')
#             IEMOCAP_PARTITIONS[line[1][1:]] = line[0]

#         # print(IEMOCAP_PARTITIONS)


#         # value = att_lbl.loc[att_lbl['File_name']=='Ses05M_script03_2_M042.wav']

#         # print(value['EmoAct'].to_string(index=False))



#         CONSENSUS = {}

#         fname, emo_ft, emo_ag, emo_sd, emo_dg, emo_ex, emo_fr, emo_nt, emo_sp, emo_hp, act, val, dom, s_id, gender, split = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

#         for key in IEMOCAP_PARTITIONS.keys():
#             value_categ = categ_lbl.loc[categ_lbl['File_name']==key]

#             used = value_categ['Used'].to_string(index=False)

#             if used == '1':
#                 value_attr = att_lbl.loc[att_lbl['File_name']==key]
#                 fname.append(key)
#                 emo_ft.append(value_categ['frustrated'].to_string(index=False))
#                 emo_ag.append(value_categ['angry'].to_string(index=False))
#                 emo_sd.append(value_categ['sad'].to_string(index=False))
#                 emo_dg.append(value_categ['disgust'].to_string(index=False))
#                 emo_ex.append(value_categ['excited'].to_string(index=False))
#                 emo_fr.append(value_categ['fear'].to_string(index=False))
#                 emo_nt.append(value_categ['neutral'].to_string(index=False))
#                 emo_sp.append(value_categ['surprise'].to_string(index=False))
#                 emo_hp.append(value_categ['happy'].to_string(index=False))
#                 act.append(value_attr['EmoAct'].to_string(index=False))
#                 val.append(value_attr['EmoVal'].to_string(index=False))
#                 dom.append(value_attr['EmoDom'].to_string(index=False))
#                 s_id.append(key[3:5])
#                 if key[-8:-7] == 'M':
#                     gender.append('Male')
#                 elif key[-8:-7] == 'F':
#                     gender.append('Female')
#                 else:
#                     print(key)
#                 split.append(IEMOCAP_PARTITIONS[key])


#         CONSENSUS['FileName'] = fname
#         CONSENSUS['frustrated'] = emo_ft
#         CONSENSUS['angry'] = emo_ag
#         CONSENSUS['sad'] = emo_sd
#         CONSENSUS['disgust'] = emo_dg
#         CONSENSUS['excited'] = emo_ex
#         CONSENSUS['fear'] = emo_fr
#         CONSENSUS['neutral'] = emo_nt
#         CONSENSUS['surprise'] = emo_sp
#         CONSENSUS['happy'] = emo_hp
#         CONSENSUS['EmoAct'] = act
#         CONSENSUS['EmoVal'] = val
#         CONSENSUS['EmoDom'] = dom
#         CONSENSUS['SpkrID'] = s_id
#         CONSENSUS['Gender'] = gender
#         CONSENSUS['Split_Set'] = split

#         print(max(dom), max(val), max(act))
#         print(min(dom), min(val), min(act))

#         # print(CONSENSUS)


#         df = pd.DataFrame(CONSENSUS)

#         name = dir_to_save + '/labels_consensus_' + number + '.csv'
#         # print(CONSENSUS)
        
#         df.to_csv(name,index=False)

