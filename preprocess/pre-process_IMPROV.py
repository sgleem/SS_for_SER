#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:24:13 2021

@author: lucas
"""
from distutils import file_util
import numpy as np
import os.path
import os 
from glob import glob
import pandas as pd
#%% Frames Extraction
from scipy.io import savemat
# folds = ['session1', 'session2', 'session3','session4', 'session5', 'session6']
# loc = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/Audio/'
# typ = ['P', 'R', 'S', 'T']

# parts = {}

# for f in folds:
#     temp = []
#     sess = loc + f + '/'
#     # print(sess)
#     for e in sorted(os.listdir(sess)):
#         utters = sess + e + '/'
    
#         for t in sorted(os.listdir(utters)):
#             directory = utters + t
#             vids = sorted(os.walk(directory + '/'))
#             for vid in vids[0][2]:
#                 temp.append(vid)
#                 # print(vid)
#     parts[f] = temp


# print(parts.keys())



# # filename = 'partitions_1.txt'

# # train = ['session5', 'session6','session1','session2']
# # test = ['session4']
# # development = ['session3']

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

# values_emo = ['D', 'M', 'P']

# for val_emo in values_emo: 

#     e_file = 'labels_consensus_EmoS_4class_' + val_emo
#     emo_file = e_file + '.csv'

#     dir_to_save = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/labels_processed_david_latest/Partioned_data_secondary/'+ e_file

#     os.mkdir(dir_to_save)
#     file_loc = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/Evalution.txt'
#     categorical_emo = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/labels_processed_david_latest/Secondary_Emotion/' + emo_file
#     label_consensus_att = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/labels_processed_david_latest/Attributes/labels_consensus_attributes.csv'


#     att_lbl = pd.read_csv (label_consensus_att)
#     categ_lbl = pd.read_csv (categorical_emo)

#     file1 = open(file_loc, 'r')
#     Lines = file1.readlines()
    
#     count = 0
#     # Strips the newline character
#     IMPROV_LABELS = {}
#     NAN_list = []

#     for line in Lines:
#         line = line.strip()

#         if line[0:3] =='UTD':
#             line = line.replace('UTD','MSP')
#             line = line.replace('.avi','.wav')
#             # print(line[0:30],line[32:33],line[35:45],line[47:57],line[59:69])
#             try:
#                 IMPROV_LABELS[line[0:30]] = [line[32:33],float(line[37:45]),float(line[49:57]),float(line[61:69])]
#                 # print(IMPROV_LABELS[line[0:30]])
#             except:
#                 NAN_list.append(line[0:30])

#     print(len(NAN_list), len(IMPROV_LABELS.keys()))

#     numbers = ['1','2','3','4','5','6']
#     for number in numbers:
#         partition = 'partitions_' + number + '.txt'

#         file_loc = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/partitions/' + partition

#         file_part = open(file_loc, 'r')
#         Lines_part = file_part.readlines()
        
#         # Strips the newline character
#         IMPROV_PARTITIONS = {}
#         NAN_list = []

#         for line in Lines_part:
#             line = line.strip().split(';')
#             IMPROV_PARTITIONS[line[1][1:]] = line[0]

#         # print(IMPROV_LABELS)


#         CONSENSUS = {}

#         fname, emo_a, emo_s, emo_n, emo_h, act, val, dom, s_id, gender, split = [],[],[],[],[],[],[],[],[],[],[]

#         for key in IMPROV_LABELS.keys():
#             # print(key)
#             value_categ = categ_lbl.loc[categ_lbl['File_name']==key[4:]]

#             used = value_categ['Used'].to_string(index=False)
#             used = str(int(used))
#             if used == '1':
#                 value_attr = att_lbl.loc[att_lbl['File_name']==key[4:]]
#                 fname.append(key)
#                 emo_a.append(value_categ['angry'].to_string(index=False))
#                 emo_s.append(value_categ['sad'].to_string(index=False))
#                 emo_n.append(value_categ['neutral'].to_string(index=False))
#                 emo_h.append(value_categ['happy'].to_string(index=False))
#                 act.append(value_attr['EmoAct'].to_string(index=False))
#                 val.append(value_attr['EmoVal'].to_string(index=False))
#                 dom.append(value_attr['EmoDom'].to_string(index=False))
#                 s_id.append(key[17:19])
#                 if key[16:17] == 'M':
#                     gender.append('Male')
#                 elif key[16:17] == 'F':
#                     gender.append('Female')
#                 split.append(IMPROV_PARTITIONS[key])


#         CONSENSUS['FileName'] = fname
#         CONSENSUS['angry'] = emo_a
#         CONSENSUS['sad'] = emo_s
#         CONSENSUS['neutral'] = emo_n
#         CONSENSUS['happy'] = emo_h
#         CONSENSUS['EmoAct'] = act
#         CONSENSUS['EmoVal'] = val
#         CONSENSUS['EmoDom'] = dom
#         CONSENSUS['SpkrID'] = s_id
#         CONSENSUS['Gender'] = gender
#         CONSENSUS['Split_Set'] = split


#         print(max(dom), max(val), max(act))
#         print(min(dom), min(val), min(act))


#         df = pd.DataFrame(CONSENSUS)

#         name = dir_to_save + '/labels_consensus_' + number + '.csv'
#         # print(CONSENSUS)

#         df.to_csv(name,index=False)


#%% ALL EMOTIONS
folds = ['session1', 'session2', 'session3','session4', 'session5', 'session6']
loc = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/Audio/'
typ = ['P', 'R', 'S', 'T']

parts = {}

for f in folds:
    temp = []
    sess = loc + f + '/'
    # print(sess)
    for e in sorted(os.listdir(sess)):
        utters = sess + e + '/'
    
        for t in sorted(os.listdir(utters)):
            directory = utters + t
            vids = sorted(os.walk(directory + '/'))
            for vid in vids[0][2]:
                temp.append(vid)
                # print(vid)
    parts[f] = temp


print(parts.keys())



# filename = 'partitions_1.txt'

# train = ['session5', 'session6','session1','session2']
# test = ['session4']
# development = ['session3']

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

values_emo = ['D', 'M', 'P']

for val_emo in values_emo: 

    e_file = 'labels_consensus_EmoS_10class_' + val_emo
    e_file2 = 'labels_consensus_EmoS_ALLclass_' + val_emo
    emo_file = e_file + '.csv'

    dir_to_save = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/All_Emotions/'+ e_file2

    os.mkdir(dir_to_save)
    file_loc = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/Evalution.txt'
    categorical_emo = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/All_Emotions/' + emo_file
    label_consensus_att = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/labels_processed_david_latest/Attributes/labels_consensus_attributes.csv'


    att_lbl = pd.read_csv (label_consensus_att)
    categ_lbl = pd.read_csv (categorical_emo)

    file1 = open(file_loc, 'r')
    Lines = file1.readlines()
    
    count = 0
    # Strips the newline character
    IMPROV_LABELS = {}
    NAN_list = []

    for line in Lines:
        line = line.strip()

        if line[0:3] =='UTD':
            line = line.replace('UTD','MSP')
            line = line.replace('.avi','.wav')
            # print(line[0:30],line[32:33],line[35:45],line[47:57],line[59:69])
            try:
                IMPROV_LABELS[line[0:30]] = [line[32:33],float(line[37:45]), \
                    float(line[49:57]),float(line[61:69])]
                # print(IMPROV_LABELS[line[0:30]])
            except:
                NAN_list.append(line[0:30])

    print(len(NAN_list), len(IMPROV_LABELS.keys()))

    numbers = ['1','2','3','4','5','6']
    for number in numbers:
        partition = 'partitions_' + number + '.txt'

        file_loc = '/home/podcast/Desktop/MSP-Hackathon/MSP-IMPROV_processing/partitions/' + partition

        file_part = open(file_loc, 'r')
        Lines_part = file_part.readlines()
        
        # Strips the newline character
        IMPROV_PARTITIONS = {}
        NAN_list = []

        for line in Lines_part:
            line = line.strip().split(';')
            IMPROV_PARTITIONS[line[1][1:]] = line[0]

        # print(IMPROV_LABELS)


        CONSENSUS = {}

        fname, emo_d, emo_f, emo_a, emo_s, emo_dg, emo_ex, emo_fr, emo_nt, emo_sp, emo_h, act, val, dom, s_id, gender, split = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

        for key in IMPROV_LABELS.keys():
            # print(key)
            value_categ = categ_lbl.loc[categ_lbl['File_name']==key[4:]]

            used = value_categ['Used'].to_string(index=False)
            used = str(int(used))
            if used == '1':
                value_attr = att_lbl.loc[att_lbl['File_name']==key[4:]]
                fname.append(key)
                emo_d.append(value_categ['depressed'].to_string(index=False))
                emo_f.append(value_categ['frustrated'].to_string(index=False))
                emo_a.append(value_categ['angry'].to_string(index=False))
                emo_s.append(value_categ['sad'].to_string(index=False))
                emo_dg.append(value_categ['disgust'].to_string(index=False))
                emo_ex.append(value_categ['excited'].to_string(index=False))
                emo_fr.append(value_categ['fear'].to_string(index=False))
                emo_nt.append(value_categ['neutral'].to_string(index=False))
                emo_sp.append(value_categ['surprise'].to_string(index=False))
                emo_h.append(value_categ['happy'].to_string(index=False))
                act.append(value_attr['EmoAct'].to_string(index=False))
                val.append(value_attr['EmoVal'].to_string(index=False))
                dom.append(value_attr['EmoDom'].to_string(index=False))
                s_id.append(key[17:19])
                if key[16:17] == 'M':
                    gender.append('Male')
                elif key[16:17] == 'F':
                    gender.append('Female')
                split.append(IMPROV_PARTITIONS[key])


        CONSENSUS['FileName'] = fname
        CONSENSUS['depressed'] = emo_d
        CONSENSUS['frustrated'] = emo_f
        CONSENSUS['angry'] = emo_a
        CONSENSUS['sad'] = emo_s       
        CONSENSUS['disgust'] = emo_dg
        CONSENSUS['excited'] = emo_ex
        CONSENSUS['fear'] = emo_fr
        CONSENSUS['neutral'] = emo_nt
        CONSENSUS['surprise'] = emo_sp
        CONSENSUS['happy'] = emo_h
        CONSENSUS['EmoAct'] = act
        CONSENSUS['EmoVal'] = val
        CONSENSUS['EmoDom'] = dom
        CONSENSUS['SpkrID'] = s_id
        CONSENSUS['Gender'] = gender
        CONSENSUS['Split_Set'] = split


        print(max(dom), max(val), max(act))
        print(min(dom), min(val), min(act))


        df = pd.DataFrame(CONSENSUS)

        name = dir_to_save + '/labels_consensus_' + number + '.csv'
        # print(CONSENSUS)

        df.to_csv(name,index=False)
