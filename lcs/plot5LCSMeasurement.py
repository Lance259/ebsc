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
from localFunctions import deinterleave, gradient_to_dirac_scaled, support, find_first_occurrence_index, calc_sample_mse, calc_rmtd_lmtd
    
import pandas as pd
import seaborn as sns
########### USER PARAMETERS #################    
    
    
samplerate = 192000
block_size = 512
num_streams = 6
num_levels = 5

filepath_measurements = "Measurement4.bin"
freq_of_meausred_signal = 100 #in Hertz

plot_periods = 2

no_of_crossings_for_interp = 500

#for point-wise mse calculation at sample instances: 
skip_first_n_samples = 10


gradient_thresh = 0.005 # where to detect a sample, based on the height of the gradient

max_level = 5.0
levels = [1.0, 2.0, 2.5, 3.0, 4.0]

# calculated parameters
samples_per_period = int(samplerate / freq_of_meausred_signal)
no_of_samples_for_mse = no_of_crossings_for_interp

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
fig.suptitle('Measurement 17Hz sine, full scale (5V p2p)')
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
mse_calc_start_index = find_first_occurrence_index(support_integral, 1)
mse_calc_end_index = find_first_occurrence_index(support_integral, no_of_crossings_for_interp) + 1

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


############## calculate error metrics ###############
# calculate the mse at sample - instances
sample_mse = []
rmtd_mean_arr = []
lmtd_mean_arr= []

no_det = []
time_deviations_list = []


orig_signal = data_deinterleaved_normalized[0][mse_calc_start_index : mse_calc_end_index]

for i in range(5):
    sample_mse.append(calc_sample_mse(data_deinterleaved_normalized[0], level_crossings_scaled[i], skip_first_n_samples, no_of_samples_for_mse))
    rmtd, lmtd, not_detectable = calc_rmtd_lmtd(orig_signal, level_crossings_scaled[i][mse_calc_start_index : mse_calc_end_index], t0=int(samples_per_period/8), y_tol=0.005, us_factor=16, returnArray=True)

    for r in rmtd:
        time_deviations_list.append(dict([('time deviation / ms', 1e3*r/samplerate), ('direction', 'rmtd'), ('level', i+1)]))
    for l in lmtd:
        time_deviations_list.append(dict([('time deviation / ms', 1e3*l/samplerate), ('direction', 'lmtd'), ('level', i+1)]))
        
    rmtd_mean_arr.append((rmtd.mean(), rmtd.std()))
    lmtd_mean_arr.append((lmtd.mean(), lmtd.std()))
    no_det.append(not_detectable)
    
error_metrics_each_level = np.array([sample_mse, np.array(rmtd_mean_arr)[:,0]/samplerate,np.array(rmtd_mean_arr)[:,1]/samplerate, np.array(lmtd_mean_arr)[:,0]/samplerate ,np.array(lmtd_mean_arr)[:,1]/samplerate, no_det])
# reference error metrics: [mse, rmtd_mean, rmtd_std, lmtd_mean, lmtd_std, no_undetectable]

error_metrics_average = [q.mean(axis=0) for q in error_metrics_each_level]

time_deviations = pd.DataFrame(time_deviations_list, columns = ['time deviation / ms', 'direction', 'level'])
plt.figure('lmtd_rmtd plot')
sns.barplot(time_deviations, x="time deviation / ms", y="level", hue="direction", orient='h')

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
