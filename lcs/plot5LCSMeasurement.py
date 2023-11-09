#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:40:29 2023

@author: lhinz
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write


def deinterleave(data, num_streams, block_size):
    blocks = np.reshape(data, (-1, block_size))
    print(blocks.shape)
    output = np.zeros((num_streams, int(len(data)/num_streams)))
    
    for i in range(num_streams):
        output[i,:] = blocks[i::num_streams].flatten()
        
    return output

def gradient_to_dirac(grad_signal, grad_thresh, amp):
    #edge_detect_kernel = np.array([-1, 2, -1])
    #output = np.zeros(len(grad_signal))
    edges = np.convolve((np.abs(grad_signal) > grad_thresh).astype(int), np.array([1,-1]), mode="same") 
    return (edges > 0).astype(int)*amp

    
    
    
    
samplerate = 192000
block_size = 512
num_streams = 6

plot_start_s = 0.34
plot_end_s = 0.4

gradient_thresh = 0.01 # where to detect a sample, based on the height of the gradient

levels = [1.0, 2.0, 2.5, 3.0, 4.0]





filepath_measurements = "Measurement2.bin"

raw_data = np.fromfile(filepath_measurements, dtype=np.float32)

data_deinterleaved = deinterleave(raw_data, num_streams, block_size)

level_crossings_scaled = np.zeros((len(levels), len(data_deinterleaved[0])))
for i, level in enumerate(levels):
    level_crossings_scaled[i] = gradient_to_dirac(np.gradient(data_deinterleaved[i+1]), gradient_thresh, level)
   
    
    
# plotting original signal
plt.figure("original signals")
for i in range(1,6):
    plt.plot((data_deinterleaved[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'input ' + str(i))
plt.plot(data_deinterleaved[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
plt.legend()


# plotting gradients of digital output (edge detection)
plt.figure("gradients")
for i in range(1,6):
    plt.plot(np.gradient(data_deinterleaved[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'input ' + str(i))
plt.plot(data_deinterleaved[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
plt.legend()

# plotting scaled diracs where the gradients are
plt.figure("scaled_diracs")
for i in range(5):
    plt.plot(level_crossings_scaled[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label = 'input ' + str(i))
plt.plot(data_deinterleaved[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
plt.legend()
#plt.plot(data_deinterleaved[4])
#plt.plot(np.gradient(data_deinterleaved[4]))
#%%

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write


filepath_measurements = "/home/lhinz/Seafile/MasterS3/MasterProject/ebsc/lcs/inputSignal.bin"
raw_data = np.fromfile(filepath_measurements, dtype=np.float32)

new_data = np.reshape(raw_data, (-1,512))
print(new_data.shape)
new_new_data = new_data[1::2]
plt.plot(new_new_data.flatten())