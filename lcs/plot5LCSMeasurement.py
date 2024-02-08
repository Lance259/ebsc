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
from localFunctions import deinterleave, gradient_to_dirac_scaled, support, find_first_occurrence_index, calc_sample_mse, calc_rmtd_lmtd, calc_error_metrics
    
import pandas as pd
import seaborn as sns
########### USER PARAMETERS #################    
    
    
samplerate = 192000
block_size = 512
num_streams = 6
num_levels = 5

filepath_measurements = "Measurement5.bin"
freq_of_meausred_signal = 500 #in Hertz
titlestring = '500Hz sine, full scale (5V p2p)'


plot_periods = 2

no_of_samples_for_mse = 500

#for point-wise mse calculation at sample instances: 
skip_first_n_samples = 10


gradient_thresh = 0.005 # where to detect a sample, based on the height of the gradient

max_level = 5.0
levels = [1.0, 2.0, 2.5, 3.0, 4.0]

# calculated parameters

plot_start_s = (skip_first_n_samples/(num_levels * 2)) / freq_of_meausred_signal # we expect num_levels*2 samples per period
plot_end_s = plot_start_s + plot_periods / freq_of_meausred_signal
########################################

plt.close('all')



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
#fig.suptitle('Measurement Signals: ' + str(filepath_measurements))
fig.suptitle('Measurement ' + titlestring)
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
    ax[1].plot(x_plot, 100*np.gradient(data_deinterleaved[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
ax[1].plot(x_plot, data_deinterleaved_normalized[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[1].legend(loc='lower left')
ax[1].set_ylabel('amplitude')
ax[1].set_title('Gradients (scaled)')
# plotting scaled diracs where the gradients are

for i in range(5):
    ax[2].plot(x_plot, level_crossings_scaled[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label = 'in ' + str(i+1))
ax[2].plot(x_plot, data_deinterleaved_normalized[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[3].set_xlabel('s')
ax[2].set_ylabel('amplitude')
ax[2].set_title('Event-based samples')
ax[2].legend(loc='lower left')
#plt.plot(data_deinterleaved[4])
#plt.plot(np.gradient(data_deinterleaved[4]))





####################### 
"""
scaled_diracs = np.zeros(level_crossings_scaled[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)].shape)
for i in range(5):
    scaled_diracs += level_crossings_scaled[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]


plt.figure()
plt.plot(scaled_diracs)
plt.plot(orig_signal)

corr = np.correlate(scaled_diracs, orig_signal, mode='full')
plt.plot(corr)

idx = find_first_occurrence_index(corr, corr.max())
#print(idx)

shift = idx - len(orig_signal) 

shifted_scaled_diracs = np.roll(scaled_diracs, shift)

plt.figure()
plt.plot(scaled_diracs)
plt.plot(orig_signal)

corr2 = np.correlate(shifted_scaled_diracs, orig_signal, mode="full")
#print(find_first_occurrence_index(corr2, corr2.max()))


rmtd, lmtd, nd = calc_rmtd_lmtd(scaled_diracs, orig_signal, t0=1000)
"""
########################



# calculate the linear interpolation and continuous mse

lcs_output_combined = np.zeros(level_crossings_scaled[0].shape)
for level_crossing in level_crossings_scaled:
    lcs_output_combined += level_crossing
    
support_integral = np.cumsum((lcs_output_combined != 0).astype(int))
mse_calc_start_index = find_first_occurrence_index(support_integral, skip_first_n_samples)
mse_calc_end_index = find_first_occurrence_index(support_integral, no_of_samples_for_mse + skip_first_n_samples) + 1

# get only the samples, not the whole vector including zeros at non-sample times
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



"""
# detail examination of lcs output -> gradients -> samples 
plt.figure()
i = 1
plt.plot(x_plot, data_deinterleaved_normalized[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
plt.plot(x_plot, (data_deinterleaved_normalized[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
plt.plot(x_plot, np.gradient(data_deinterleaved[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'grad_in ' + str(i))
plt.plot(x_plot, level_crossings_scaled[i-1][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label = 'lcs_in ' + str(i))
plt.legend()
"""

orig_signal = data_deinterleaved_normalized[0][mse_calc_start_index : mse_calc_end_index]
scaled_diracs = level_crossings_scaled[:, mse_calc_start_index : mse_calc_end_index]

error_metrics_average, error_metrics_each_level = calc_error_metrics(orig_signal, scaled_diracs, samplerate, freq_of_meausred_signal, True, titlestring)


plt.figure('Signal and samples')
for i in range(5):
    plt.plot(x_plot, level_crossings_scaled[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label = 'in ' + str(i+1))
plt.plot(x_plot, data_deinterleaved_normalized[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
plt.xlabel('s')
plt.ylabel('amplitude')
plt.title('Event-based samples ' + titlestring)
plt.legend(loc='lower left')

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
