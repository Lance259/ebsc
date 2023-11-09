#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 10:16:40 2023

@author: lhinz
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write

samplerate_orig = 360
samplerate_new = 192000
sample_length_s = 7
filepath_resampled = "AB_222_242908_245427_192k.bin"
filepath_orig = "AB_222_242908_245427_360.bin"
filepath_sine = "sine.bin"

ekg_data = np.genfromtxt("ekgData/survey_data_7s/AB_222_242908_245427.csv", delimiter=",")


mlii_data = ekg_data[:,0][~np.isnan(ekg_data[:,0])]
while (len(mlii_data) < sample_length_s*samplerate_orig) :
    mlii_data = np.append(mlii_data, [0])
t_orig = np.linspace(0, sample_length_s, len(mlii_data))




mlii_data_resampled = scipy.signal.resample(mlii_data, int(len(mlii_data)/samplerate_orig*samplerate_new))
t_resampled = np.linspace(0, sample_length_s, len(mlii_data_resampled))

plt.plot(t_orig[0:40], mlii_data[0:40], t_resampled[0:25000], mlii_data_resampled[0:25000])


np.float32(mlii_data_resampled).tofile(filepath_resampled)
np.float32(mlii_data).tofile(filepath_orig)

sine_data = np.sin(2*np.pi*1000*t_resampled)
np.float32(sine_data).tofile(filepath_sine)



#%% 

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write

samplerate_orig = 360
samplerate_new = 44100
sample_length_s = 7
filepath_resampled = "AB_222_242908_245427_192k.bin"
filepath_orig = "AB_222_242908_245427_360.bin"
filepath_sine = "sine.bin"

ekg_data = np.genfromtxt("ekgData/survey_data_7s/AB_222_242908_245427.csv", delimiter=",")


mlii_data = ekg_data[:,0][~np.isnan(ekg_data[:,0])]
while (len(mlii_data) < sample_length_s*samplerate_orig) :
    mlii_data = np.append(mlii_data, [0])
t_orig = np.linspace(0, sample_length_s, len(mlii_data))




mlii_data_resampled = scipy.signal.resample(mlii_data, int(len(mlii_data)/samplerate_orig*samplerate_new))
t_resampled = np.linspace(0, sample_length_s, len(mlii_data_resampled))

#plt.plot(t_orig[0:40], mlii_data[0:40], t_resampled[0:25000], mlii_data_resampled[0:25000])

sine_data = np.sin(2*np.pi*1000*t_resampled)
np.float32(sine_data).tofile(filepath_sine)

rate = 44100
scaled = np.int16(mlii_data_resampled  / np.max(np.abs(mlii_data_resampled )) * 32767)
plt.plot(t_resampled, mlii_data_resampled)
#write('test.wav', rate, scaled)

# linear interpolation for upsampling (very slow)

"""
upsampled_arr = np.repeat(mlii_data, samplerate_new)

x = np.linspace(0, sample_length_s, len(mlii_data))
x_upsampled = np.linspace(0, sample_length_s, len(upsampled_arr))

# Interpolate upsampled array
interpolated_arr = np.interp(x_upsampled, x, mlii_data)

resampled_data = interpolated_arr.reshape(-1, samplerate_orig).mean(axis=1)

print(resampled_data)
"""
#plt.figure()
#plt.plot(upsampled_arr)
#plt.plot(interpolated_arr)