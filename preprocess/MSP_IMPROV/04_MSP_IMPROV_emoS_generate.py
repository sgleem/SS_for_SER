#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 12:27:34 2022

@author: david
"""

import pandas as pd
import numpy as np


def check_emo(each_emo):
    Emotion_count_dic = {'Depressed':0,'Frustrated':0,'Angry':0,'Sad':0,'Disgusted':0,
                         'Excited':0,'Fear':0,'Neutral':0,'Surprised':0,'Happy':0,'Other':0}

    if each_emo in ["NeutralConcerned","Neutralrelaxed","Neutralindifferent","Neutralapologetic"
                    ,"Neutralinterested","Neutraldoubtful","Sadconfused","Neutralcalm"
                    ,"Neutralpassinganacquaintance","NeutralResigned","NeutralTired"
                    ,"NeutralQuestioning","Neutralcomfortable","NeutralSerious"
                    ,"NeutralFlat","NeutralThisclipshowsConfusion"
                    ,"Neutralinquisitive","NeutralInquisitive","Neutralcurious"
                    ,"NeutralConcern","NeutralWorried","NeutralHopeful"
                    ,"Neutralsarcastic","NeutralBored","Neutralnonchalance"
                    ,"NeutralBored","Neutralconfused","NeutralSleepy"
                    ,"Neutralunconcerned","Neutralinquiring","Neutralproud"
                    ,"Neutralconcerned","Neutralirritability"
                    ]:
        all_content = ['Neutral','Other']
    elif each_emo in ["Neutral(clipcutoffearly?)"]:
        all_content = ['Neutral']                 
    elif each_emo in ["Happycheerful","HappyAnimated","HappyOptimistic"
                      ,"HappySelf-consciousness","Happyreleived"
                      ,"Happyabrupt","HappyTalkative","HappySexy","Happyamused"]:
        all_content = ['Happy','Other']     
    elif each_emo == "AngryAbrupt":
        all_content = ['Angry','Other']  
    elif each_emo in ["Fearful","Feart"]:
        all_content = ['Fear']             
    elif each_emo in ["FearfulConcern","Fearfulworry"]:
        all_content = ['Fear','Other']   
    elif each_emo in ["Sadconfused","Sadglum","Sadconcern","SadDisappointed"
                      ,"SadDefeated","Saddejected","SadApologetic","Sadconcerned"
                      ,"Sadindifferent","Sadapologetic"]:
        all_content = ['Sad','Other']  
    elif each_emo in ["Exciteduncertainty","ExcitedDistressed","Excitedanxious"
                      ,"ExcitedTheclipisaccusatory","ExcitedCocky","Excitedsore"
                      ,"ExcitedScared","Exciteditsstuck","Excitedhopeful"
                      ,"ExcitedHumorous","ExcitedProud","Excitednervous","Excitedestatic"
                      ,"Excitedplayful","ExcitedGiggly","Excitedfoolingaround"
                      ,"Excitedrelief","Excitedshy","Excitedannoyed"]:
        all_content = ['Excited','Other']  
    elif each_emo == "Surprise":
        all_content = ['Surprised']  
    elif each_emo in ["Surprisedconfused","SurprisedDisbelieving","SurprisedGrateful"]:
        all_content = ['Surprised',"Other"]  
    elif each_emo == "AngryThisclipshowsauthority":
        all_content = ['Angry',"Other"]  
    elif each_emo in ["Frustratedimpatient","Frustratedconfused","FrustratedDisappointed"
                      ,"Frustratedrartional","Frustrateddisappiontmentwithhimself"
                      ,"Frustratedresignation","Frustrateditsstuck"
                      ]:
        all_content = ['Frustrated',"Other"]  
    elif each_emo == "Excitement":
        all_content = ['Excited'] 
    elif each_emo in ["confused","interested","annoyed"]:
        all_content = ['Other'] 
    elif each_emo in ["DisgustedThisclipshowsCalmness","Disgustedirritated","Disgustedaccusing"
                      ,"Disgustedannoyed","Disgustedstifled"
                      ]:
        all_content = ['Disgusted',"Other"]           
    elif each_emo in ["DepressedDissapointed","Depressedthevideoitsstuck",
                      "Depressedexasperated","Depressedboredom","Depressedtearful"
                      ,"Depressedexasperated","Depressedunhappy","Depressedregretful"
                      ,"Depressedupset","Depressedconfused","Depressedapologetic"
                      ]:
        all_content = ['Depressed',"Other"]    
    elif each_emo in ["Disgust","DisgustedD"]:
        all_content = ['Disgusted']  
    elif each_emo == "thereisnoclipdisplaying":
        all_content = []
    elif 'Other' in each_emo:
        all_content = ['Other']  
    elif each_emo in ["Other(Thisclipshowsnervousness)","sympathy","perplexed","nervous"
                      ,"Amused","assured","Annoyed","forceful","concern","Thisclipshowsboredomandapathy"
                      ,"apathetic","hopeful","Curious","Bored","Playful","concerned","insecure","Embarassed"
                      ,"puzzled","seemsunsure","flustered","Confused","Nervous"
                      ]:
        all_content = ['Other']  
    elif each_emo == "Sad,Surprise":
        all_content = ['Sad','Surprised']  
    elif each_emo in list(Emotion_count_dic.keys()):
        all_content = [each_emo]
    else:
        if ')' not in each_emo:
            if each_emo != " ":
                if "," in each_emo:
                    all_content = each_emo.replace("Surprise","Surprised").split(",")
                else:
                    # print("&"*10)
                    # print(each_emo)
                    # print("!"*10)
                    all_content = []
            else:
                all_content = []
        else:
            all_content = []
    return all_content

def get_all(input_lst):
    Emotion_count_dic = {'Depressed':0,'Frustrated':0,'Angry':0,'Sad':0,'Disgusted':0,
                         'Excited':0,'Fear':0,'Neutral':0,'Surprised':0,'Happy':0,'Other':0}
    
    emotion_list = list(input_lst)[0]
    # print(emotion_list)
    All_emotion_list = []
    All_final_list = []
    for each_emo in emotion_list:
        #print(each_emo)
        if "|" in each_emo:
            all_content = each_emo.split("|")
            # temp_one = []
            # for each_one in all_content:
            #     temp_one.extend(each_one.split(","))
            # all_content = temp_one
        elif "," in each_emo:
            all_content = each_emo.split(",")
        else:
            all_content = check_emo(each_emo)
        All_emotion_list.extend(all_content)
    # print(All_emotion_list)   
    for each_right_emo in All_emotion_list:
        chected_emo = check_emo(each_right_emo)
        All_final_list.extend(chected_emo)
    
    for each_good in All_final_list:
        if each_good =="Excitement":
            each_good = "Excited"
        if each_good =="Feart":
            each_good = "Fear"
        if each_good in Emotion_count_dic:
            Emotion_count_dic[each_good]+=1
        else:
            print(each_good)
    # print(All_final_list,len(All_final_list))      
    
    return  np.array(list(Emotion_count_dic.values()))
    

#%%
want = ['EmoS']
data = pd.read_pickle('./output/Evalution_raw.pkl')
data_want = data[want]

#%%
['Neutral,Frustrated,Surprised', 
 'Frustrated', 'Neutral', 
 'Angry,Frustrated,Surprised,Disgusted,Other(showsContempt)', 
 'Angry,Frustrated,Disgusted', 
 'Angry,Disgusted']
# All_Dataframe = data_want.apply(get_all,axis=1)
All_Dataframe = data_want.apply(get_all,axis=1)


aa_all = []
aa = np.asarray(All_Dataframe.to_numpy())

for each in aa:
    aa_all.append(each)
    
aa_all = np.asarray(aa_all)    

All_Dataframe_final = pd.DataFrame(aa_all,index=data_want.index,columns=['depressed', 'frustrated', 'angry', 'sad', 'disgust', 'excited', 'fear', 'neutral', 'surprise', 'happy', 'other'])
All_Dataframe_final.index.name = 'File_name'
All_Dataframe_final.to_csv('./output/Secondary_Emotion_class_raw_count.csv')
All_Dataframe_final_sum = All_Dataframe_final.sum(axis=0)
All_Dataframe_final_sum.sort_values(0,ascending=False).plot.bar(title="MSP-IMPROV Secondary Emotion",legend=0)





