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
from scipy.fft import fft

samplerate_orig = 360
samplerate_new = 192000
sample_length_s = 7
n_fft = 8*2048
filepath_ekg = '/home/lhinz/Seafile/MasterS3/MasterProject/ebsc/lcs/ekgData/survey_data_7s/N_101_187_2706.csv'


ekg_data = np.genfromtxt(filepath_ekg, delimiter=",")


mlii_data = ekg_data[:,0][~np.isnan(ekg_data[:,0])]
while (len(mlii_data) < sample_length_s*samplerate_orig) :
    mlii_data = np.append(mlii_data, [0])
t_orig = np.linspace(0, sample_length_s, len(mlii_data))


mlii_data = mlii_data - mlii_data.mean()

Mlii_data = fft(mlii_data, n=n_fft)/n_fft

f = samplerate_orig * np.arange(0,len(Mlii_data),1) / len(Mlii_data)

#plt.plot(mlii_data)
fig, ax = plt.subplots(2,1,sharex=False)
plt.rcParams['text.usetex'] = True
ax[0].plot(t_orig[:1000], mlii_data[:1000])
ax[0].set_xlabel('$t$/s')
ax[0].set_ylabel('s(t)')
ax[0].set_title('EGK Data in time domain')

ax[1].semilogy(f[:int(n_fft/2)], abs(Mlii_data[:int(n_fft/2)]))
ax[1].set_xlabel('$f$/Hz')
ax[1].set_ylabel('|S(f)|')
ax[1].set_title('EGK Data in frequency domain')


"""
mlii_data_resampled = scipy.signal.resample(mlii_data, int(len(mlii_data)/samplerate_orig*samplerate_new))
t_resampled = np.linspace(0, sample_length_s, len(mlii_data_resampled))

plt.plot(t_orig[0:40], mlii_data[0:40], t_resampled[0:25000], mlii_data_resampled[0:25000])


np.float32(mlii_data_resampled).tofile(filepath_resampled)
np.float32(mlii_data).tofile(filepath_orig)

sine_data = np.sin(2*np.pi*1000*t_resampled)
np.float32(sine_data).tofile(filepath_sine)
"""


#%% 

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write

samplerate_orig = 360
samplerate_new = 44100
sample_length_s = 7
directory_in_str = "ekgData/survey_data_7s"
directory_out_str = "ekgData_wav"
#filepath_resampled = "AB_222_242908_245427_192k.bin"
#filepath_sine = "sine.bin"


for file in os.listdir(directory_in_str):
     filename = os.fsdecode(file)
     if filename.endswith(".csv"): 
         # print(os.path.join(directory_in_str, filename))
        
        ekg_data = np.genfromtxt(os.path.join(directory_in_str, filename), delimiter=",")
        
        
        # get the mlii values as array from the csv file, without NaNs
        mlii_data = ekg_data[:,0][~np.isnan(ekg_data[:,0])]
        
        #append some zeros to make the array length fit the sample length
        while (len(mlii_data) < sample_length_s*samplerate_orig) :
            mlii_data = np.append(mlii_data, [0])
        t_orig = np.linspace(0, sample_length_s, len(mlii_data))
        
        # resample the signal, so that we have now an array of length sample_length_s * samplerate_new
        mlii_data_resampled = scipy.signal.resample(mlii_data, int(samplerate_new * len(mlii_data)/samplerate_orig))
        t_resampled = np.linspace(0, sample_length_s, len(mlii_data_resampled))
        
        #plt.plot(t_orig[0:40], mlii_data[0:40], t_resampled[0:25000], mlii_data_resampled[0:25000])
        
        """
        # create a sine as test
        sine_data = np.sin(2*np.pi*1000*t_resampled)
        np.float32(sine_data).tofile(filepath_sine)
        """
        
        rate_wav = samplerate_new
        # scale to 16Bit wav resolution
        scaled_16Bit = np.int16(mlii_data_resampled  / np.max(np.abs(mlii_data_resampled )) * 32767)
        write(os.path.join(directory_out_str, filename.replace('.csv','.wav')), rate_wav, scaled_16Bit)





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