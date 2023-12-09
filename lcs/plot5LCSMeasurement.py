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
from scipy.interpolate import interp1d


def deinterleave(data, num_streams, block_size):
    blocks = np.reshape(data, (-1, block_size))
    print(blocks.shape)
    output = np.zeros((num_streams, int(len(data)/num_streams)))
    
    for i in range(num_streams):
        output[i,:] = blocks[i::num_streams].flatten()
        
    return output

def gradient_to_dirac_scaled(grad_signal, grad_thresh, amp):
    #edge_detect_kernel = np.array([-1, 2, -1])
    #output = np.zeros(len(grad_signal))
    edges = np.convolve((np.abs(grad_signal) > grad_thresh).astype(int), np.array([1,-1]), mode="same") 
    return (edges > 0).astype(int)*amp

    
def support(vec):
    return (vec != 0).astype(int).sum()

def find_first_occurrence_index(a, b):
    """
    finds the index of array a where value b first occurs
    """
    # Convert the input list to a numpy array
    a = np.array(a)
    
    # Find the indices where the value is equal to b
    indices = np.where(a == b)[0]
    
    # Return the index of the first occurrence
    if len(indices) > 0:
        return indices[0]
    else:
        # Handle the case where b is not in the array
        return None
    
########### USER PARAMETERS #################    
    
    
samplerate = 192000
block_size = 512
num_streams = 6

plot_start_s = 0.2
plot_end_s = 0.34

no_of_crossings_for_interp = 1000

gradient_thresh = 0.005 # where to detect a sample, based on the height of the gradient

max_level = 5.0
levels = [1.0, 2.0, 2.5, 3.0, 4.0]

########################################

plt.close('all')

filepath_measurements = "Measurement6.bin"

raw_data = np.fromfile(filepath_measurements, dtype=np.float32)

data_deinterleaved = deinterleave(raw_data, num_streams, block_size)
x_values = np.arange(0,len(data_deinterleaved[0]),1)
level_crossings_scaled = np.zeros((len(levels), len(data_deinterleaved[0])))
for i, level in enumerate(levels):
    level_crossings_scaled[i] = gradient_to_dirac_scaled(np.gradient(data_deinterleaved[i+1]), gradient_thresh, level)
   
    

#normalizing signals
scalefactor = (max_level/2) / np.max(np.abs(data_deinterleaved[0]))
    
data_deinterleaved_normalized = data_deinterleaved * scalefactor
data_deinterleaved_normalized[0] += max_level/2

# plotting original signal

fig, ax = plt.subplots(4,1,sharex=True)
fig.suptitle('Measurement Signals: ' + str(filepath_measurements))
fig.canvas.manager.set_window_title('plot_' + str(filepath_measurements).replace('.bin', ''))
x_plot = x_values[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)] / samplerate
for i in range(1,6):
    ax[0].plot(x_plot, (data_deinterleaved_normalized[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
ax[0].plot(x_plot, data_deinterleaved_normalized[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[0].legend(loc='lower left')
ax[0].set_ylabel('amplitude')
ax[0].set_title('Original Signals')

# plotting gradients of digital output (edge detection)

for i in range(1,6):
    ax[1].plot(x_plot, np.gradient(data_deinterleaved[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
ax[1].plot(x_plot, data_deinterleaved[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[1].legend(loc='lower left')
ax[1].set_ylabel('amplitude')
ax[1].set_title('Gradients')
# plotting scaled diracs where the gradients are

for i in range(5):
    ax[2].plot(x_plot, level_crossings_scaled[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label = 'in ' + str(i+1))
ax[2].plot(x_plot, data_deinterleaved_normalized[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[3].set_xlabel('s')
ax[2].set_ylabel('amplitude')
ax[2].set_title('Scaled Signal and Samples')
ax[2].legend(loc='lower left')
#plt.plot(data_deinterleaved[4])
#plt.plot(np.gradient(data_deinterleaved[4]))



lcs_output_combined = np.zeros(level_crossings_scaled[0].shape)
for level_crossing in level_crossings_scaled:
    lcs_output_combined += level_crossing
    
support_integral = np.cumsum((lcs_output_combined != 0).astype(int))
mse_calc_start_index = find_first_occurrence_index(support_integral, 1)
mse_calc_end_index = find_first_occurrence_index(support_integral, no_of_crossings_for_interp) + 1


lcs_output_combined_2 = lcs_output_combined[lcs_output_combined != 0]
x_values_2 = x_values[lcs_output_combined != 0]

f = interp1d(x_values_2, lcs_output_combined_2, kind='linear')

x_values_for_interp = x_values[mse_calc_start_index:mse_calc_end_index]
lcs_output_interpolated = f(x_values_for_interp)

plt_idx_start = int(samplerate*plot_start_s) - mse_calc_start_index
plt_idx_end = int(samplerate*plot_end_s) - mse_calc_start_index
#ax[3].figure('Reconstruction with linear interpolation')
ax[3].plot(x_values_for_interp[plt_idx_start:plt_idx_end]/samplerate, lcs_output_interpolated[plt_idx_start:plt_idx_end], label='linear interpolation')
ax[3].plot(x_values_for_interp[plt_idx_start:plt_idx_end]/samplerate, data_deinterleaved_normalized[0][int(samplerate*plot_start_s):int(samplerate*plot_end_s)], label="orig signal")

mse_vec = ((lcs_output_interpolated - data_deinterleaved_normalized[0][mse_calc_start_index:mse_calc_end_index])**2)
ax[3].plot(x_values_for_interp[plt_idx_start:plt_idx_end]/samplerate, mse_vec[plt_idx_start:plt_idx_end], label='mse: {:.3f}'.format(mse_vec.mean()) )
ax[3].legend(loc='lower left')
ax[3].set_ylabel('amplitude')
ax[3].set_title('Reconstruction with linear interpolation')
print(mse_vec.mean())


plt.figure()
i = 2
plt.plot(x_plot, data_deinterleaved_normalized[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
plt.plot(x_plot, (data_deinterleaved_normalized[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
plt.plot(x_plot, np.gradient(data_deinterleaved[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'grad_in ' + str(i))
plt.plot(x_plot, level_crossings_scaled[i-1][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label = 'lcs_in ' + str(i+1))
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