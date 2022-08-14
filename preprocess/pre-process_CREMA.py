# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Feb 10 14:24:13 2021

# @author: lucas
# """
# from distutils import file_util
# import numpy as np
# import os.path
# import os 
# from glob import glob
# import pandas as pd



# number = '5'



# demographics_file = '/home/podcast/Desktop/MSP-Hackathon/CREMA-D/VideoDemographics.csv'
# demographics_info = pd.read_csv(demographics_file)

# speakers = demographics_info['ActorID'].to_list()
# speakers.sort()

# partitions_len = len(speakers)//5

# partitions = {}

# partitions['speakers_1'] = speakers[:partitions_len]
# partitions['speakers_2'] = speakers[partitions_len:2*partitions_len]
# partitions['speakers_3'] = speakers[2*partitions_len:3*partitions_len]
# partitions['speakers_4'] = speakers[3*partitions_len:4*partitions_len]
# partitions['speakers_5'] = speakers[4*partitions_len:]

# # print(partitions)


# train = partitions['speakers_2'] +  partitions['speakers_3'] +  partitions['speakers_4']
# test = partitions['speakers_5']
# development = partitions['speakers_1']


# name_of_file = 'label_consensus_' + number + '.csv'

# labels = ['labels_consensus_4class_D', 'labels_consensus_4class_M', 'labels_consensus_4class_P']

# for label in labels:

#     dir_to_save = '/home/podcast/Desktop/MSP-Hackathon/CREMA-D/Partitioned_data_CREMA-D/' + label + '/'

#     categorical_emo = '/home/podcast/Desktop/MSP-Hackathon/CREMA-D/Emotion/' + label + '.csv'

#     categ_lbl = pd.read_csv(categorical_emo)


#     CONSENSUS = {}

#     fname, emo_a, emo_s, emo_n, emo_h, s_id, gender, split = [],[],[],[],[],[],[],[],

#     parts = [train,test,development]

#     count = 0

#     for part in parts:
#         count += 1
#         keys = categ_lbl['File_name'].to_list()
#         for key in keys:
#             if int(key[:4]) in part: 
#                 value_categ = categ_lbl.loc[categ_lbl['File_name']==key]
#                 used = value_categ['Used'].to_string(index=False)
#                 used = str(int(used))
#                 value_sex = demographics_info.loc[demographics_info['ActorID']==int(key[:4])]
#                 if used == '1':
#                     fname.append(key)
#                     emo_a.append(value_categ['angry'].to_string(index=False))
#                     emo_s.append(value_categ['sad'].to_string(index=False))
#                     emo_n.append(value_categ['neutral'].to_string(index=False))
#                     emo_h.append(value_categ['happy'].to_string(index=False))
#                     s_id.append(key[:4])
                    
#                     gender.append(value_sex['Sex'].to_string(index=False))

#                     if count == 1:
#                         split.append('Train')
#                     elif count == 2:
#                         split.append('Test')
#                     elif count == 3:
#                         split.append('Development')


#     CONSENSUS['FileName'] = fname
#     CONSENSUS['angry'] = emo_a
#     CONSENSUS['sad'] = emo_s
#     CONSENSUS['neutral'] = emo_n
#     CONSENSUS['happy'] = emo_h
#     CONSENSUS['SpkrID'] = s_id
#     CONSENSUS['Gender'] = gender
#     CONSENSUS['Split_Set'] = split


#     df = pd.DataFrame(CONSENSUS)

#     name = dir_to_save + name_of_file
#     # print(CONSENSUS)

#     df.to_csv(name,index=False)















#%% ALL CLASSES

from distutils import file_util
import numpy as np
import os.path
import os 
from glob import glob
import pandas as pd



number = '1'



demographics_file = '/home/podcast/Desktop/MSP-Hackathon/CREMA-D/VideoDemographics.csv'
demographics_info = pd.read_csv(demographics_file)

speakers = demographics_info['ActorID'].to_list()
speakers.sort()

partitions_len = len(speakers)//5

partitions = {}

partitions['speakers_1'] = speakers[:partitions_len]
partitions['speakers_2'] = speakers[partitions_len:2*partitions_len]
partitions['speakers_3'] = speakers[2*partitions_len:3*partitions_len]
partitions['speakers_4'] = speakers[3*partitions_len:4*partitions_len]
partitions['speakers_5'] = speakers[4*partitions_len:]

# print(partitions)


train = partitions['speakers_3'] +  partitions['speakers_4'] +  partitions['speakers_5']
test = partitions['speakers_1']
development = partitions['speakers_2']


name_of_file = 'label_consensus_' + number + '.csv'

labels = ['labels_consensus_6class_D', 'labels_consensus_6class_M', 'labels_consensus_6class_P']

for label in labels:

    dir_to_save = '/home/podcast/Desktop/MSP-Hackathon/CREMA-D/Partitioned_data_CREMA-D/' + label + '/'

    categorical_emo = '/home/podcast/Desktop/MSP-Hackathon/CREMA-D/All_Emotions/' + label + '.csv'

    categ_lbl = pd.read_csv(categorical_emo)


    CONSENSUS = {}

    fname, emo_a, emo_s, emo_d, emo_f, emo_n, emo_h, s_id, gender, split = [],[],[],[],[],[],[],[],[],[]

    parts = [train,test,development]

    count = 0

    for part in parts:
        count += 1
        keys = categ_lbl['File_name'].to_list()
        for key in keys:
            if int(key[:4]) in part: 
                value_categ = categ_lbl.loc[categ_lbl['File_name']==key]
                used = value_categ['Used'].to_string(index=False)
                used = str(int(used))
                value_sex = demographics_info.loc[demographics_info['ActorID']==int(key[:4])]
                if used == '1':
                    fname.append(key)
                    emo_a.append(value_categ['angry'].to_string(index=False))
                    emo_s.append(value_categ['sad'].to_string(index=False))
                    emo_d.append(value_categ['disgust'].to_string(index=False))
                    emo_f.append(value_categ['fear'].to_string(index=False))
                    emo_n.append(value_categ['neutral'].to_string(index=False))
                    emo_h.append(value_categ['happy'].to_string(index=False))
                    s_id.append(key[:4])
                    
                    gender.append(value_sex['Sex'].to_string(index=False))

                    if count == 1:
                        split.append('Train')
                    elif count == 2:
                        split.append('Test')
                    elif count == 3:
                        split.append('Development')


    CONSENSUS['FileName'] = fname
    CONSENSUS['angry'] = emo_a
    CONSENSUS['sad'] = emo_s
    CONSENSUS['disgust'] = emo_d
    CONSENSUS['fear'] = emo_f
    CONSENSUS['neutral'] = emo_n
    CONSENSUS['happy'] = emo_h
    CONSENSUS['SpkrID'] = s_id
    CONSENSUS['Gender'] = gender
    CONSENSUS['Split_Set'] = split


    df = pd.DataFrame(CONSENSUS)

    name = dir_to_save + name_of_file
    # print(CONSENSUS)

    df.to_csv(name,index=False)