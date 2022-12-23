#!/usr/bin/env python
# coding: utf-8

# In[147]:


import os
import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing import MinMaxScaler
import scipy as sp
import numpy as np
import math
import mne


# In[148]:


num_elect_avail={"study01":57,
                 "study02":63,
                 "study03":63,
                 "study04":61,
                 "study05":34}


# In[149]:


# study_part_map={}
# whole_data=r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\wholedata"
# for r,d,f in os.walk(whole_data):
#     for file in f:
#         if '.set' in file:
#             temp=file.split('_')
#             participant_id=temp[1]
#             study=int(temp[0][-1])
#             study_part_map[file]=study
            


# In[150]:


# study_part_map


# In[151]:


# preprocessed_path = r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\preprocessed\"
# file_names = []
# for i in os.listdir(path):
#     file_names.append(os.path.join(path,i))


# In[152]:


# file_names


# In[153]:


def get_fft(snippet):
    Fs = 128.0  # sampling rate
    snippet_time = len(snippet) / Fs
    Ts = 1.0 / Fs  # sampling interval
    t = np.arange(0, snippet_time, Ts)  # time vector

    y = snippet
    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range #shows negative and positive frequency components
    frq = frq[range(n // 2)]  # one side frequency range

    Y = np.fft.fft(y) / n  # fft computing and normalization
    Y = Y[range(n // 2)]
    return frq, abs(Y)


def theta_alpha_beta_averages(f, Y):
    theta_range = (4, 8)
    alpha_range = (8, 12)
    beta_range = (12, 40)
    theta = Y[(f > theta_range[0]) & (f <= theta_range[1])].mean()
    alpha = Y[(f > alpha_range[0]) & (f <= alpha_range[1])].mean()
    beta = Y[(f > beta_range[0]) & (f <= beta_range[1])].mean()
    return theta, alpha, beta


def make_steps(samples, frame_duration, overlap):
    Fs = 128.0
    i = 0
    intervals = []
    samples_per_frame = Fs * frame_duration
    while i + samples_per_frame <= samples:
        intervals.append((i, i + samples_per_frame))
        i = i + samples_per_frame - int(samples_per_frame * overlap)
#     print(intervals)
    return intervals


def make_frames(df, frame_duration):
    Fs = 128.0
    frame_length = Fs * frame_duration
    frames = []
#     print(len(df))
    steps = make_steps(len(df), frame_duration, overlap)
    for i, _ in enumerate(steps):
        frame = []
        if i == 0:
            continue
        else:

            for channel in df.columns:
#                 print(channel,)
                snippet = np.array(df.loc[steps[i][0]:steps[i][1], int(channel)])
                f, Y = get_fft(snippet)
                theta, alpha, beta = theta_alpha_beta_averages(f, Y)
                frame.append([theta, alpha, beta])
#                 print(theta)
            

        # frames.append(list(np.array(frame).T.flatten()))
        frames.append(frame)
#     print((np.array(frames)).shape)
    return np.array(frames)



def cartesian_to_spherical(x, y, z):
    x2_y2 = x ** 2 + y ** 2
    radius = math.sqrt(x2_y2 + z ** 2)
    elevation = math.atan2(z, math.sqrt(x2_y2))
    azimuth = math.atan2(y, x)
    return radius, elevation, azimuth


def polar_to_cartesian(theta, rho):
    return rho * math.cos(theta), rho * math.sin(theta)


def azimuthal_projection(position):
    [radius, elevation, azimuth] = cartesian_to_spherical(position[0], position[1], position[2])
    return polar_to_cartesian(azimuth, math.pi / 2 - elevation)




# In[154]:


def elec_location(locs):
    locations_2d = []
    for position in locs:
        locations_2d.append(azimuthal_projection(position))
    locations_2d = np.array(locations_2d)
    return locations_2d

electrode_data_path=r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\electrode_data"
locs_path1=os.path.join(electrode_data_path,"ourelectpos0.npy")
locs_path2=os.path.join(electrode_data_path,"ourelectpos1.npy")
locs_path3=os.path.join(electrode_data_path,"ourelectpos2.npy")
locs_path4=os.path.join(electrode_data_path,"ourelectpos3.npy")
locs_path5=os.path.join(electrode_data_path,"ourelectpos4.npy")

locs1=np.load(locs_path1)
locs2=np.load(locs_path2)
locs3=np.load(locs_path3)
locs4=np.load(locs_path4)
locs5=np.load(locs_path5)


# In[155]:


locs_2d1 = elec_location(locs1)
locs_2d2 = elec_location(locs2)
locs_2d3 = elec_location(locs3)
locs_2d4 = elec_location(locs4)
locs_2d5 = elec_location(locs5)


# In[156]:


locs_2d_dict={"study01":locs_2d1,
                 "study02":locs_2d2,
                 "study03":locs_2d3,
                 "study04":locs_2d4,
                 "study05":locs_2d5}


# In[157]:



frame_duration = 1.0
overlap = 0.94


def compute_interpolation_weights(electrode_locations, grid):
    triangulation = sp.spatial.Delaunay(electrode_locations)
    simplex = triangulation.find_simplex(grid)
    vertices = np.take(triangulation.simplices, simplex, axis=0)
    temp = np.take(triangulation.transform, simplex, axis=0)
    delta = grid - temp[:, 2]
    bary = np.einsum('njk,nk->nj', temp[:, :2, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, interpolation_weights, fill_value=np.nan):
    # import pdb;pdb.set_trace()
    vtx, wts = interpolation_weights
#     print("len(vtx) : ",len(vtx))
#     print("shape.values : ",values.shape)
    output = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    output[np.any(wts < 0, axis=1)] = fill_value
    return output


def convert_to_images(features, n_gridpoints, n_channels, channel_size, interpolation_weights):
                      #(112,57,3),  32, 3, 57,
    n_samples = features.shape[0]
    # import pdb;pdb.set_trace()
    
    interpolated_data = np.zeros([n_channels, n_samples, n_gridpoints * n_gridpoints])  #3,112, 32*32

    for channel in range(n_channels):
        frequency_features = features[:, channel * channel_size: (channel + 1) * channel_size]
#         print(frequency_features.shape)
        for sample in range(n_samples):
            interpolated_data[channel, sample, :] = interpolate(frequency_features[sample], interpolation_weights)
    return interpolated_data.reshape((n_channels, n_samples, n_gridpoints, n_gridpoints))


# In[158]:



videopath=r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\video"
if not os.path.exists(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\video"):
    os.makedirs(r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\video")
    


# In[162]:


preprocessed_path = r"C:\Users\DELL\Desktop\20credit\EEG_eye_artifact\preprocessed"
Fs = 128.0
frame_length = Fs*frame_duration
X_1 = []
for r,d,f in os.walk(preprocessed_path):
    for study in d :
        if "study" in study:
            folderpath=os.path.join(videopath,study)
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            locs_2d=locs_2d_dict[study]
            for r1,d1,f2 in os.walk(os.path.join(preprocessed_path,study)):
                for participant in d1:
                    participantpath=os.path.join(folderpath,participant)
                    if not os.path.exists(participantpath):
                        os.makedirs(participantpath)
                    for r2,d3,files in os.walk(os.path.join(r1,participant)):
                      file_names=[]
                      for x in files:
                        file_names.append(os.path.join(r2,x))
                      for i,file in enumerate(file_names):
                        print('Processing session: ', file, '. (', i+1, ' of ', len(file_names), ')')
                        data = np.load(file)  # genfromtxt(file, delimiter=',')
#                         print(data.shape)
                        df = pd.DataFrame(data)
                        df = df/10e-06
                        df=df.T
                        steps = np.arange(0,len(df),frame_length)
                        X_0 = make_frames(df,frame_duration)
#                         print(df.shape)
#                         print(X_0.shape) # (112,57)
#                         print(X_0)
                        window_size = num_elect_avail[study]*3 # no. electrodes * frequency bands (3)
                        X_1 = X_0.reshape(len(X_0),window_size)
#                         print(X_1)
#                         print(X_1.shape)   #(112,171)
                        n_frequencies = 3
                        n_gridpoints = 32
                        features = np.array(X_1)
#                         print(features)
#                         print(features.shape)
                        n_windows = features.shape[1] // window_size #7  (171 // 171 == 1)
                        channel_size = window_size // n_frequencies #64  (171 // 3 == 57 )

                        grid_x, grid_y = np.mgrid[min((np.array(locs_2d))[:, 0]):max((np.array(locs_2d))[:, 0]):n_gridpoints * 1j,
                                    min((np.array(locs_2d))[:, 1]):max((np.array(locs_2d))[:, 1]):n_gridpoints * 1j]
                        grid = np.zeros([n_gridpoints * n_gridpoints, 2])
#                         print(grid.shape)
                        
                        grid[:, 0] = grid_x.flatten()
                        grid[:, 1] = grid_y.flatten()
                        interpolation_weights = compute_interpolation_weights((np.array(locs_2d)), grid)  #tuple of two elements.
                        
                        average_features = np.split(features[:, :], n_windows, axis=1)
                        average_features = np.sum(average_features, axis=0)
                        average_features = average_features / n_windows
#                         print(n_gridpoints, n_frequencies, channel_size)
                        images = convert_to_images(features, n_gridpoints, n_frequencies, channel_size, interpolation_weights)
#                         print(images)
                        images = images[np.newaxis,...]
                        images = np.transpose(images, (0, 2, 3, 4, 1))
#                         print(images)
                        fname = file.split('\\')[-1]
#                         print(fname)
#                         print(images.shape)
#                         print(np.nan_to_num(images))
                        np.save(os.path.join(participantpath, fname), np.nan_to_num(images))

