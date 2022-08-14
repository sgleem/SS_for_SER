# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Feb 10 14:24:13 2021

# @author: lucas
# """

# #%%CREMA-D Frame extraction
# from distutils import file_util
# import numpy as np
# import os.path
# import os 
# from glob import glob
# import pandas as pd
# from scipy.io import savemat

# label_consensus_loc = '/home/podcast/Desktop/MSP-Hackathon/MSP-PODCAST-Publish-1.10/labels_consensus.csv'
# att_lbl = pd.read_csv (label_consensus_loc)
# keys = list_of_single_column = att_lbl['FileName'].tolist()

# emotionals = ['Secondary_Emotion']

# values_emo = ['D','M', 'P']



# for emotional in emotionals:

#     name_folder = 'Partitioned_data_' + emotional

#     dir_saving = '/home/podcast/Desktop/MSP-Hackathon/MSP-PODCAST-Publish-1.10/' + name_folder

#     # os.mkdir(name_folder)

#     for val_emo in values_emo: 

#         e_file = 'labels_consensus_Emo' + emotional[0] + '_4class_' + val_emo
#         emo_file = e_file + '.csv'


#         categ_labels = '/home/podcast/Desktop/MSP-Hackathon/MSP-PODCAST-Publish-1.10/labels_processed/' + emotional + '/' + emo_file
        
#         dir_to_save = dir_saving + '/' + e_file

#         os.mkdir(dir_to_save)

#         categ_lbl = pd.read_csv(categ_labels)

#         # print(categ_lbl)


#         CONSENSUS = {}

#         fname, emo_a, emo_s, emo_n, emo_h, act, val, dom, s_id, gender, split = [],[],[],[],[],[],[],[],[],[],[]

#         for key in keys:
#             value_categ = categ_lbl.loc[categ_lbl['File_name']==key]

#             used = value_categ['Used'].to_string(index=False)

#             if used == '1':
#                 value_attr = att_lbl.loc[att_lbl['FileName']==key]
#                 fname.append(key)
#                 emo_a.append(value_categ['angry'].to_string(index=False))
#                 emo_s.append(value_categ['sad'].to_string(index=False))
#                 emo_n.append(value_categ['neutral'].to_string(index=False))
#                 emo_h.append(value_categ['happy'].to_string(index=False))
#                 act.append(value_attr['EmoAct'].to_string(index=False))
#                 val.append(value_attr['EmoVal'].to_string(index=False))
#                 dom.append(value_attr['EmoDom'].to_string(index=False))
#                 s_id.append(value_attr['SpkrID'].to_string(index=False))
#                 gender.append(value_attr['Gender'].to_string(index=False))
#                 split.append(value_attr['Split_Set'].to_string(index=False))


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

#         # print(CONSENSUS)


#         df = pd.DataFrame(CONSENSUS)

#         name = dir_to_save + '/labels_consensus.csv'
#         # print(CONSENSUS)
        
#         df.to_csv(name,index=False)

# #%% 8 classs




# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Feb 10 14:24:13 2021

# @author: lucas
# """

# #%%CREMA-D Frame extraction
# from distutils import file_util
# import numpy as np
# import os.path
# import os 
# from glob import glob
# import pandas as pd
# from scipy.io import savemat

# label_consensus_loc = '/home/podcast/Desktop/MSP-Hackathon/MSP-PODCAST-Publish-1.10/labels_consensus.csv'
# att_lbl = pd.read_csv (label_consensus_loc)
# keys = list_of_single_column = att_lbl['FileName'].tolist()

# emotionals = ['Primary_Emotion']

# values_emo = ['D','M', 'P']



# for emotional in emotionals:

#     name_folder = 'Partitioned_data_' + emotional + 'ALLclass'

#     dir_saving = '/home/podcast/Desktop/MSP-Hackathon/MSP-PODCAST-Publish-1.10/' + name_folder

#     os.mkdir(name_folder)

#     for val_emo in values_emo: 

#         e_file = 'labels_consensus_Emo' + emotional[0] + '_8class_' + val_emo
#         emo_file = e_file + '.csv'


#         categ_labels = '/home/podcast/Desktop/MSP-Hackathon/MSP-PODCAST-Publish-1.10/All_Emotions/' + emo_file
        
#         dir_to_save = dir_saving + '/' + e_file

#         os.mkdir(dir_to_save)

#         categ_lbl = pd.read_csv(categ_labels)

#         # print(categ_lbl)


#         CONSENSUS = {}

#         fname, emo_a, emo_s, emo_d, emo_c,emo_f, emo_n, emo_sp, emo_h, act, val, dom, s_id, gender, split = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

#         for key in keys:
#             value_categ = categ_lbl.loc[categ_lbl['File_name']==key]

#             used = value_categ['Used'].to_string(index=False)

#             if used == '1':
#                 value_attr = att_lbl.loc[att_lbl['FileName']==key]
#                 fname.append(key)
#                 emo_a.append(value_categ['angry'].to_string(index=False))
#                 emo_s.append(value_categ['sad'].to_string(index=False))
#                 emo_d.append(value_categ['disgust'].to_string(index=False))
#                 emo_c.append(value_categ['contempt'].to_string(index=False))
#                 emo_f.append(value_categ['fear'].to_string(index=False))
#                 emo_n.append(value_categ['neutral'].to_string(index=False))
#                 emo_sp.append(value_categ['surprise'].to_string(index=False))
#                 emo_h.append(value_categ['happy'].to_string(index=False))
#                 act.append(value_attr['EmoAct'].to_string(index=False))
#                 val.append(value_attr['EmoVal'].to_string(index=False))
#                 dom.append(value_attr['EmoDom'].to_string(index=False))
#                 s_id.append(value_attr['SpkrID'].to_string(index=False))
#                 gender.append(value_attr['Gender'].to_string(index=False))
#                 split.append(value_attr['Split_Set'].to_string(index=False))


#         CONSENSUS['FileName'] = fname
#         CONSENSUS['angry'] = emo_a
#         CONSENSUS['sad'] = emo_s
#         CONSENSUS['disgust'] = emo_d
#         CONSENSUS['contempt'] = emo_c
#         CONSENSUS['fear'] = emo_f
#         CONSENSUS['neutral'] = emo_n
#         CONSENSUS['surprise'] = emo_sp
#         CONSENSUS['happy'] = emo_h
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

#         name = dir_to_save + '/labels_consensus.csv'
#         # print(CONSENSUS)
        
#         df.to_csv(name,index=False)







# #%% 16 classs




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
from scipy.io import savemat

label_consensus_loc = '/home/podcast/Desktop/MSP-Hackathon/MSP-PODCAST-Publish-1.10/labels_consensus.csv'
att_lbl = pd.read_csv (label_consensus_loc)
keys = list_of_single_column = att_lbl['FileName'].tolist()

emotionals = ['Secondary_Emotion']

values_emo = ['D','M', 'P']



for emotional in emotionals:

    name_folder = 'Partitioned_data_' + emotional + 'ALLclass'

    dir_saving = '/home/podcast/Desktop/MSP-Hackathon/MSP-PODCAST-Publish-1.10/' + name_folder

    os.mkdir(name_folder)

    for val_emo in values_emo: 

        e_file = 'labels_consensus_Emo' + emotional[0] + '_16class_' + val_emo
        emo_file = e_file + '.csv'


        categ_labels = '/home/podcast/Desktop/MSP-Hackathon/MSP-PODCAST-Publish-1.10/All_Emotions/' + emo_file
        
        dir_to_save = dir_saving + '/' + e_file

        os.mkdir(dir_to_save)

        categ_lbl = pd.read_csv(categ_labels)

        # print(categ_lbl)


        CONSENSUS = {}

        fname, emo_ag, emo_ft, emo_an, emo_ds,emo_sd, emo_dg, emo_dp, emo_ct, emo_cf, emo_cc, emo_fr, emo_nt,emo_sp, emo_am, emo_ex, emo_hp, act, val, dom, s_id, gender, split = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

        for key in keys:
            value_categ = categ_lbl.loc[categ_lbl['File_name']==key]

            used = value_categ['Used'].to_string(index=False)

            if used == '1':
                value_attr = att_lbl.loc[att_lbl['FileName']==key]
                fname.append(key)
                emo_ag.append(value_categ['angry'].to_string(index=False))
                emo_ft.append(value_categ['frustrated'].to_string(index=False))
                emo_an.append(value_categ['annoyed'].to_string(index=False))
                emo_ds.append(value_categ['disappointed'].to_string(index=False))
                emo_sd.append(value_categ['sad'].to_string(index=False))
                emo_dg.append(value_categ['disgust'].to_string(index=False))
                emo_dp.append(value_categ['depressed'].to_string(index=False))
                emo_ct.append(value_categ['contempt'].to_string(index=False))
                emo_cf.append(value_categ['confused'].to_string(index=False))
                emo_cc.append(value_categ['concerned'].to_string(index=False))
                emo_fr.append(value_categ['fear'].to_string(index=False))
                emo_nt.append(value_categ['neutral'].to_string(index=False))
                emo_sp.append(value_categ['surprise'].to_string(index=False))
                emo_am.append(value_categ['amused'].to_string(index=False))
                emo_ex.append(value_categ['excited'].to_string(index=False))
                emo_hp.append(value_categ['happy'].to_string(index=False))
                act.append(value_attr['EmoAct'].to_string(index=False))
                val.append(value_attr['EmoVal'].to_string(index=False))
                dom.append(value_attr['EmoDom'].to_string(index=False))
                s_id.append(value_attr['SpkrID'].to_string(index=False))
                gender.append(value_attr['Gender'].to_string(index=False))
                split.append(value_attr['Split_Set'].to_string(index=False))


        CONSENSUS['FileName'] = fname
        CONSENSUS['angry'] = emo_ag
        CONSENSUS['frustrated'] = emo_ft
        CONSENSUS['annoyed'] = emo_an
        CONSENSUS['disappointed'] = emo_ds
        CONSENSUS['sad'] = emo_sd
        CONSENSUS['disgust'] = emo_dg
        CONSENSUS['depressed'] = emo_dp
        CONSENSUS['contempt'] = emo_ct
        CONSENSUS['confused'] = emo_cf
        CONSENSUS['concerned'] = emo_cc
        CONSENSUS['fear'] = emo_fr
        CONSENSUS['neutral'] = emo_nt
        CONSENSUS['surprise'] = emo_sp
        CONSENSUS['amused'] = emo_am
        CONSENSUS['excited'] = emo_ex
        CONSENSUS['happy'] = emo_hp
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

        name = dir_to_save + '/labels_consensus.csv'
        # print(CONSENSUS)
        
        df.to_csv(name,index=False)

