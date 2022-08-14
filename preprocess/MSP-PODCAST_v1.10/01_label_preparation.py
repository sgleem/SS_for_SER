# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 13:34:13 2022

@author: User
"""
from collections import defaultdict
import pandas as pd
import joblib

def ddict():
    return defaultdict(ddict)

def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)

#%%
path = './labels_detailed.csv'
label_DF = pd.read_csv(path)

Primary_dict = defaultdict(dict)
Seconday_dict = defaultdict(dict)
for i in range(len(label_DF)):
# for i in range(100):
    each_record = label_DF.iloc[i]
    key  = each_record["FileName"]
    detail = each_record["EmoDetail"].replace(" ","").split("A:")[0].split(";")
    woker = detail[0]
    primary_emo = detail[1]
    secondary_emo = "".join(detail[2:])
    if primary_emo not in secondary_emo:
        secondary_emo = secondary_emo + "," + primary_emo
    
    Primary_dict[key][woker] = primary_emo
    Seconday_dict[key][woker] = secondary_emo
    
primary_dict = ddict2dict(Primary_dict)       
second_dict = ddict2dict(Seconday_dict)    
    
#%%
primary_save_path = './output/primary_dict_v110.pkl'
secondary_save_paht = './output/seconday_dict_v110.pkl'

joblib.dump(primary_dict, primary_save_path,compress=9,protocol=4)
joblib.dump(second_dict, secondary_save_paht,compress=9,protocol=4)
