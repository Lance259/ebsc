
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from localFunctions import gradient_to_dirac, find_first_occurrence_index, state_transition, state_to_lcs, extract_single_state_transition, calc_error_metrics
#%matplotlib inline
%matplotlib qt
    
filename_csv = '/home/lhinz/Seafile/MasterS3/MasterProject/ebsc/lcs/lcs_meas_data/scope_altLvls_1_3V_EKG_AFIB_202_411724_414243_Input_Vor_Preamp.csv'
titlestring = 'EKG_AFIB_202_411724_414243'
freq_of_meausred_signal = 10
switch_order_of_inputs = False

plot_start_s = 1
plot_end_s = 3

no_of_samples_for_mse = 100
skip_first_n_crossings = 20 # skip first crossings for mse calculation, because the state is not properly determined yet

gradient_thresh = 0.5 # where to detect a sample, based on the height of the gradient

#max_level = 5.0
levels = [0, 2, 2.25, 2.5, 2.75, 3, 4] #[0, 1.5, 2, 2.5, 3, 3.5, 4] # 
lcs_offset= 2.5

plt.close('all')


# get data from scope csv and clean data
scope_data_all = np.genfromtxt(filename_csv, delimiter=",")
samplerate = 1/(scope_data_all[201,0]-scope_data_all[200,0]) # calc samplerate from time data in scope csv
#scope_data = scope_data[~np.isnan(scope_data)]
scope_data = scope_data_all[200:50000, 1:5]
scope_data = np.transpose(scope_data)


scope_data_input = scope_data[0]
lcs_output_analog = np.zeros((scope_data.shape[0]-1, scope_data.shape[1]))
# some data has to be filtered, because the switching instances were not clear enough to detect state transitions
scope_data[3] = np.convolve(scope_data[3], [1,1,1], mode='same') / 3
scope_data[2] = np.convolve(scope_data[2], [1,1,1], mode='same') / 3
scope_data[1] = np.convolve(scope_data[1], [1,1,1], mode='same') / 3
scope_data[0] = np.convolve(scope_data[0], [1,1,1,1,1,1,1,1,1,1,1], mode='same') / 11
scope_data[0] -= scope_data[0].mean()


if switch_order_of_inputs:
    lcs_output_analog[0] = scope_data[3]
    lcs_output_analog[1] = scope_data[2]
    lcs_output_analog[2] = scope_data[1]    
else:
    lcs_output_analog[2] = scope_data[3]
    lcs_output_analog[1] = scope_data[2]
    lcs_output_analog[0] = scope_data[1]

# calculat gradients -> diracs (the sample instances) from the raw data
x_values = np.arange(0,scope_data.shape[1],1)
lcs_output_grads = np.zeros((scope_data.shape[0]-1, scope_data.shape[1]))
for i in range(scope_data.shape[0]-1):
    lcs_output_grads[i] = gradient_to_dirac(np.gradient(lcs_output_analog[i]), gradient_thresh, 1)
 

# decode the gray coded 3-bit signal into 6 possible states
lcs_output_state = np.zeros(len(lcs_output_grads[0,:]), dtype=np.int16)
lcs_output_state_transition = np.zeros(len(lcs_output_grads[0,:]), dtype=np.int16)
for i in range(1, len(lcs_output_grads[0,:])):
    lcs_output_state[i], lcs_output_state_transition[i] = state_transition(lcs_output_state[i-1], lcs_output_grads[:,i]) 
    
# assigning actual levels to the states 
lcs_output_levels = state_to_lcs(lcs_output_state, levels) - lcs_offset
lcs_output_diracs = state_to_lcs(lcs_output_state_transition, levels)

lcs_output_diracs_separate = np.zeros([6, len(lcs_output_state_transition)])
for i in range(1,7):
    lcs_output_diracs_separate[i-1] = state_to_lcs(extract_single_state_transition(lcs_output_state_transition, i), levels)


data_deinterleaved_normalized = scope_data


# find out the indices at the first crossing and the n-th crossing 
support_integral = np.cumsum((lcs_output_diracs != 0).astype(int))
mse_calc_start_index = find_first_occurrence_index(support_integral, skip_first_n_crossings)
try:
    mse_calc_end_index = find_first_occurrence_index(support_integral, no_of_samples_for_mse + skip_first_n_crossings) + 1
except Exception:
    print('Not enough crossings in signal, to get the desired no_of_samples_for_mse')
    mse_calc_end_index = len(support_integral) - 1000


# plotting original signal
x_plot = x_values[mse_calc_start_index:mse_calc_end_index] / samplerate
#x_plot = x_values[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)] / samplerate
n_plots = 2
n_plot_idx = 0
fig, ax = plt.subplots(n_plots,1,sharex=True)
plt.suptitle(titlestring)
#fig.suptitle('Measurement Signals: ' + str(filepath_measurements))
#fig.canvas.manager.set_window_title('plot_' + str(filepath_measurements).replace('.bin', ''))
"""
for i in range(1,4):
    ax[n_plot_idx].plot(x_plot, (scope_data[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
ax[n_plot_idx].plot(x_plot, scope_data[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[n_plot_idx].legend(loc='lower left')
ax[n_plot_idx].set_ylabel('amplitude')
ax[n_plot_idx].set_title('Original Signals')
n_plot_idx += 1

# plotting gradients of digital output (edge detection)
for i in range(1,4):
    ax[n_plot_idx].plot(x_plot, np.gradient(scope_data[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
ax[n_plot_idx].plot(x_plot, scope_data[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[n_plot_idx].legend(loc='lower left')
ax[n_plot_idx].set_ylabel('amplitude')
ax[n_plot_idx].set_title('Gradients')
n_plot_idx += 1
"""

# plotting the state

ax[n_plot_idx].plot(x_plot, scope_data[0][mse_calc_start_index:mse_calc_end_index], label="orig signal")
#ax[n_plot_idx].plot(x_plot, lcs_output_state[mse_calc_start_index:mse_calc_end_index], label="state")
#ax[n_plot_idx].plot(x_plot, lcs_output_levels[mse_calc_start_index:mse_calc_end_index], label="actual lcs levels")
ax[n_plot_idx].plot(x_plot, lcs_output_diracs[mse_calc_start_index:mse_calc_end_index] - lcs_offset , label = 'scaled diracs')
ax[n_plot_idx].set_ylabel('amplitude')
ax[n_plot_idx].set_title('Event-based states and samples')
ax[n_plot_idx].legend(loc='lower left')
#plt.plot(data_deinterleaved[4])
#plt.plot(np.gradient(data_deinterleaved[4]))
n_plot_idx += 1



# get only the data points at a crossing, without the zeros in between
lcs_output_for_interp = lcs_output_diracs[lcs_output_diracs != 0] - lcs_offset
x_values_for_interp = x_values[lcs_output_diracs != 0]

# create a function that resembles the interpolated array
f = interp1d(x_values_for_interp, lcs_output_for_interp, kind='linear')

# create the interpolated array in the interval of interest
x_values_for_interp = x_values[mse_calc_start_index:mse_calc_end_index]
lcs_output_interpolated = f(x_values_for_interp)

#ax[3].figure('Reconstruction with linear interpolation')
ax[n_plot_idx].plot(x_plot, data_deinterleaved_normalized[0][mse_calc_start_index:mse_calc_end_index], label="orig signal")

# the lcs appears to have an offset. The MSE can be lowered a lot, when we add this simple offset to the interpolated version of the signal. 
# we use a standard optimizer function from scipy to find the minimum MSE
optimized_offset = 0
"""
calc_mse = lambda offs : (((lcs_output_interpolated + offs - scope_data[0][mse_calc_start_index:mse_calc_end_index])**2)).mean()
optimized_offset = minimize(calc_mse, -0.25, method = 'Nelder-Mead').x
ax[n_plot_idx].plot(x_values_for_interp/samplerate, lcs_output_interpolated+optimized_offset, label='interp with opt. offset: ' + str(optimized_offset))
"""
mse_vec = ((lcs_output_interpolated + optimized_offset - scope_data[0][mse_calc_start_index:mse_calc_end_index])**2)
ax[n_plot_idx].plot(x_values_for_interp/samplerate, lcs_output_interpolated, label='linear interpolation')
ax[n_plot_idx].plot(x_values_for_interp/samplerate, mse_vec, label='mse: {:.3f}'.format(mse_vec.mean()) )


ax[n_plot_idx].set_ylabel('amplitude')
#ax[n_plot_idx].set_ylim([-1,2])
ax[n_plot_idx].set_title('Reconstruction with linear interpolation')
ax[n_plot_idx].legend(loc='lower left')


################ evaluation of error metrics #################
# -optimized_offset
orig_signal = + lcs_offset + scope_data[0][mse_calc_start_index : mse_calc_end_index]
scaled_diracs = lcs_output_diracs_separate[:, mse_calc_start_index : mse_calc_end_index]

#orig_signal = orig_signal[:3000]
#scaled_diracs = scaled_diracs[:,:3000]
x_plot = np.arange(0,len(orig_signal),1)/samplerate
plt.figure('exp2_signalPlot2_')
plt.title(titlestring)
plt.plot(x_plot, orig_signal)
for i in range(6):
    plt.plot(x_plot, scaled_diracs[i])
plt.xlabel('t/s')
plt.ylabel('amplitude')
error_metrics_average, error_metrics_each_level = calc_error_metrics(orig_signal, scaled_diracs, samplerate, freq_of_meausred_signal, True, titlestring)
error_metrics_average = np.array(error_metrics_average)*6/5
print('ferting')
###############################################################