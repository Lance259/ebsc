
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write
from scipy.interpolate import interp1d


def gradient_to_dirac(grad_signal, grad_thresh, amp):
    #edge_detect_kernel = np.array([-1, 2, -1])
    #output = np.zeros(len(grad_signal))
    edges = np.convolve((np.abs(grad_signal) > grad_thresh).astype(int), np.array([1,-1]), mode="same") 
    return (edges > 0).astype(int)*amp

samplerate_orig = 360
samplerate_new = 44100
sample_length_s = 7
directory_in_str = ""
filename_csv = '/home/lhinz/Seafile/MasterS3/MasterProject/ebsc/lcs/scope_triangle_1000Hz_poti40proc.csv'
directory_out_str = "ekgData_wav"
#filepath_resampled = "AB_222_242908_245427_192k.bin"
#filepath_sine = "sine.bin"

samplerate = 192000
block_size = 512
num_streams = 6

plot_start_s = 0
plot_end_s = 0.7

mse_calc_start_index = 0
mse_calc_end_index = 1000000

gradient_thresh = 0.9 # where to detect a sample, based on the height of the gradient

max_level = 5.0
levels = [2.0, 2.5, 3.0]

scope_data = np.genfromtxt(filename_csv, delimiter=",")
scope_data = scope_data[2:7600, 1:5]
data_deinterleaved = np.transpose(scope_data)

x_values = np.arange(0,len(data_deinterleaved[0]),1)
level_crossings_scaled = np.zeros((len(levels), len(data_deinterleaved[0])))
for i, level in enumerate(levels):
    level_crossings_scaled[i] = gradient_to_dirac(np.gradient(data_deinterleaved[i+1]), gradient_thresh, level)
   
    

#normalizing signals
"""
scalefactor = (max_level/2) / np.max(np.abs(data_deinterleaved[0]))
    
data_deinterleaved_normalized = data_deinterleaved * scalefactor
data_deinterleaved_normalized[0] += max_level/2
"""
data_deinterleaved_normalized = data_deinterleaved
# plotting original signal

fig, ax = plt.subplots(3,1,sharex=True)
#fig.suptitle('Measurement Signals: ' + str(filepath_measurements))
#fig.canvas.manager.set_window_title('plot_' + str(filepath_measurements).replace('.bin', ''))
for i in range(1,4):
    ax[0].plot(x_values[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], (data_deinterleaved_normalized[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
ax[0].plot(x_values[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], data_deinterleaved_normalized[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[0].legend(loc='lower left')
ax[0].set_ylabel('amplitude')
ax[0].set_title('Original Signals')

# plotting gradients of digital output (edge detection)

for i in range(1,4):
    ax[1].plot(x_values[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], np.gradient(data_deinterleaved[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
ax[1].plot(x_values[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], data_deinterleaved[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[1].legend(loc='lower left')
ax[1].set_ylabel('amplitude')
ax[1].set_title('Gradients')
# plotting scaled diracs where the gradients are

for i in range(3):
    ax[2].plot(x_values[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], level_crossings_scaled[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label = 'in ' + str(i+1))
ax[2].plot(x_values[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], data_deinterleaved_normalized[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[2].set_ylabel('amplitude')
ax[2].set_title('Scaled Signal and Samples')
ax[2].legend(loc='lower left')
#plt.plot(data_deinterleaved[4])
#plt.plot(np.gradient(data_deinterleaved[4]))



lcs_output_combined = np.zeros(level_crossings_scaled[0].shape)
for level_crossing in level_crossings_scaled:
    lcs_output_combined += level_crossing
    

lcs_output_combined_2 = lcs_output_combined[lcs_output_combined != 0]
x_values_2 = x_values[lcs_output_combined != 0]

f = interp1d(x_values_2, lcs_output_combined_2, kind='linear')

x_values_for_interp = x_values[mse_calc_start_index:mse_calc_end_index]
lcs_output_interpolated = f(x_values_for_interp)

plt_idx_start = int(samplerate*plot_start_s) - mse_calc_start_index
plt_idx_end = int(samplerate*plot_end_s) - mse_calc_start_index
#ax[3].figure('Reconstruction with linear interpolation')
print(mse_vec.mean())


#scope_data = scope_data[~np.isnan(scope_data)]


plt.plot(scope_data[:,0], scope_data[:,1])
plt.plot(scope_data[:,0], scope_data[:,2])
plt.plot(scope_data[:,0], scope_data[:,3])
plt.plot(scope_data[:,0], scope_data[:,4])


"""
# resample the signal, so that we have now an array of length sample_length_s * samplerate_new
mlii_data_resampled = scipy.signal.resample(mlii_data, int(samplerate_new * len(mlii_data)/samplerate_orig))
t_resampled = np.linspace(0, sample_length_s, len(mlii_data_resampled))

#plt.plot(t_orig[0:40], mlii_data[0:40], t_resampled[0:25000], mlii_data_resampled[0:25000])


# create a sine as test
sine_data = np.sin(2*np.pi*1000*t_resampled)
np.float32(sine_data).tofile(filepath_sine)


rate_wav = samplerate_new
# scale to 16Bit wav resolution
scaled_16Bit = np.int16(mlii_data_resampled  / np.max(np.abs(mlii_data_resampled )) * 32767)
write(os.path.join(directory_out_str, filename.replace('.csv','.wav')), rate_wav, scaled_16Bit)

"""