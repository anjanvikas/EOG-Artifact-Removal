#!/usr/bin/env python
# coding: utf-8

# In[117]:


import numpy as np
import mne
import pandas as pd
import os


# In[118]:


electrode_names_file=r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\electrode_data\10_20_elc_electrodes.txt"
electrode_pos_file=r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\electrode_data\1020_elc_position.txt"


# In[119]:


def readFile(fileName):
        fileObj = open(fileName, "r") #opens the file in read mode
        words = fileObj.read().splitlines() #puts the file into an array
        fileObj.close()
        return words


# In[120]:


electrode_names=readFile(electrode_names_file)
electrode_pos=readFile(electrode_pos_file)


# In[121]:


for i in range(len(electrode_names)):
    temp=electrode_pos[i].split(" ")
    nparray=[]
    for j in range(len(temp)):
        if temp[j]!='':
            nparray.append(float(temp[j]))  
    electrode_pos[i]=np.array(nparray)


# In[122]:


electrode_dict={}
for i in range(len(electrode_names)):
    electrode_dict[electrode_names[i]]=electrode_pos[i]


# In[123]:


study1=mne.io.read_epochs_eeglab(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\study01\study01_p01_prep.set")
study2=mne.io.read_epochs_eeglab(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\study02\study02_p02_prep.set")
study3=mne.io.read_epochs_eeglab(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\study03\study03_p03_prep.set")
study4=mne.io.read_epochs_eeglab(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\study04\study04_p03_prep.set")
study5=mne.io.read_epochs_eeglab(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\study05\study05_p06_prep.set")


# In[124]:


study1.pick('eeg')
study2.pick('eeg')
study3.pick('eeg')
study4.pick('eeg')
study5.pick('eeg')


# In[125]:


column_names1=list(study1.ch_names)
column_names2=list(study2.ch_names)
column_names3=list(study3.ch_names)
column_names4=list(study4.ch_names)
column_names5=list(study5.ch_names)


# In[126]:


ourelect_pos1=[]
ourelect_pos2=[]
ourelect_pos3=[]
ourelect_pos4=[]
ourelect_pos5=[]


# In[127]:


exclude1=[]
exclude2=[]
exclude3=[]
exclude4=[]
exclude5=[]


# In[128]:



def find_elect_pos(electrode_dict,column_names):
    ourelect_pos=[]
    exclude=[]
    for i in range(len(column_names)):
        if column_names[i] in electrode_dict:
            ourelect_pos.append(electrode_dict[column_names[i]])
        else:
            exclude.append(column_names[i])
    return ourelect_pos,exclude


# In[129]:


ourelect_pos1,exclude1=np.array(find_elect_pos(electrode_dict,column_names1))
ourelect_pos2,exclude2=np.array(find_elect_pos(electrode_dict,column_names2))
ourelect_pos3,exclude3=np.array(find_elect_pos(electrode_dict,column_names3))
ourelect_pos4,exclude4=np.array(find_elect_pos(electrode_dict,column_names4))
ourelect_pos5,exclude5=np.array(find_elect_pos(electrode_dict,column_names5))


# In[130]:


ourelect_pos=[ourelect_pos1,
             ourelect_pos2,
             ourelect_pos3,
             ourelect_pos4,
             ourelect_pos5]
exclude=[
    exclude1,
    exclude2,
    exclude3,
    exclude4,
    exclude5
]


# In[131]:


for i in range(5):
    print(len(ourelect_pos[i]),len(exclude[i]))


# In[132]:


for i in range(5):
    np.save(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\electrode_data\ourelectpos"+str(i),ourelect_pos[i])


# In[133]:


import os
path = r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\wholedata"

preprocessedpath=r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\preprocessed"
if not os.path.exists(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\preprocessed"):
    os.makedirs(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\preprocessed")
    


# In[134]:


for r,d,f in os.walk(path):
    for file in f:
        if '.set' in file:
            temp=file.split('_')
            participant_id=temp[1]
            folderpath=os.path.join(preprocessedpath,temp[0])
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            destpath=os.path.join(folderpath,participant_id)
            if not os.path.exists(destpath):
               os.makedirs(destpath)
            
            srcpath=os.path.join(path,file)
            data=mne.io.read_epochs_eeglab(srcpath)
            data.pick('eeg')
            study_num=int(temp[0][-1])
            channels=mne.pick_channels(data.ch_names, include=[],exclude=exclude[study_num-1])
            data_resampled = data.resample(128)
            data_preprocessed=mne.filter.filter_data(data_resampled.get_data(),128,1,63)
            x=data_preprocessed
            x_df=data.to_data_frame()
            labels=x_df['label']
            chn=(x.shape)[0]
            window=(x.shape)[2]
            idx=0
            for trail in range(chn):
                final_part_data=x[trail]
                final_part_data=final_part_data[channels]
                print(file,window,idx,chn,chn*window,final_part_data.shape)
                label=labels[idx]
                idx=idx+window
                np.save(os.path.join(destpath,str(trail)+'_'+str(int(label))),final_part_data)

