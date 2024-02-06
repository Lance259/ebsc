
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from localFunctions import gradient_to_dirac, find_first_occurrence_index, state_transition, state_to_lcs
#%matplotlib inline
%matplotlib qt
    

filename_csv = '/home/lhinz/Seafile/MasterS3/MasterProject/ebsc/lcs/scope_altLvls_1_3V_EKG_N_100_2225_4744_Input_Vor_Preamp.csv'

plot_start_s = 0
plot_end_s = 7

no_of_crossings_for_interp = 180
skip_first_n_crossings = 15 # skip first crossings for mse calculation, because the state is not properly determined yet

gradient_thresh = 0.8 # where to detect a sample, based on the height of the gradient

#max_level = 5.0
levels = [0, 2, 2.25, 2.5, 2.75, 3, 4] #[0, 1.5, 2, 2.5, 3, 3.5, 4] #
lcs_offset= 2.5

plt.close('all')


# get data from scope csv and clean data
scope_data_all = np.genfromtxt(filename_csv, delimiter=",")
samplerate = 1/(scope_data_all[201,0]-scope_data_all[200,0]) # calc samplerate from time data in scope csv
#scope_data = scope_data[~np.isnan(scope_data)]
scope_data = scope_data_all[200:62000, 1:5]
scope_data = np.transpose(scope_data)

scope_data_input = scope_data[0]
lcs_output_analog = np.zeros((scope_data.shape[0]-1, scope_data.shape[1]))
# some data has to be filtered, because the switching instances were not clear enough to detect state transitions
scope_data[3] = np.convolve(scope_data[3], [1,1,1], mode='same') / 3
scope_data[2] = np.convolve(scope_data[2], [1,1,1], mode='same') / 3
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


data_deinterleaved_normalized = scope_data
# plotting original signal

x_plot = x_values[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)] / samplerate
fig, ax = plt.subplots(4,1,sharex=True)
#fig.suptitle('Measurement Signals: ' + str(filepath_measurements))
#fig.canvas.manager.set_window_title('plot_' + str(filepath_measurements).replace('.bin', ''))
for i in range(1,4):
    ax[0].plot(x_plot, (scope_data[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
ax[0].plot(x_plot, scope_data[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[0].legend(loc='lower left')
ax[0].set_ylabel('amplitude')
ax[0].set_title('Original Signals')

# plotting gradients of digital output (edge detection)

for i in range(1,4):
    ax[1].plot(x_plot, np.gradient(scope_data[i][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)]), label = 'in ' + str(i))
ax[1].plot(x_plot, scope_data[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[1].legend(loc='lower left')
ax[1].set_ylabel('amplitude')
ax[1].set_title('Gradients')


# plotting the state 
ax[2].plot(x_plot, scope_data[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")
ax[2].plot(x_plot, lcs_output_state[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="state")
ax[2].plot(x_plot, lcs_output_levels[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="actual lcs levels")
ax[2].plot(x_plot, lcs_output_diracs[int(samplerate*plot_start_s) : int(samplerate*plot_end_s)] - lcs_offset , label = 'scaled diracs')
ax[2].set_ylabel('amplitude')
ax[2].set_title('Scaled Signal and Samples')
ax[2].legend(loc='lower left')
#plt.plot(data_deinterleaved[4])
#plt.plot(np.gradient(data_deinterleaved[4]))




# find out the indices at the first crossing and the n-th crossing 
support_integral = np.cumsum((lcs_output_diracs != 0).astype(int))
mse_calc_start_index = find_first_occurrence_index(support_integral, skip_first_n_crossings)
try:
    mse_calc_end_index = find_first_occurrence_index(support_integral, no_of_crossings_for_interp) + 1
except Exception:
    mse_calc_end_index = len(support_integral) - 1000



# get only the data points at a crossing, without the zeros in between
lcs_output_for_interp = lcs_output_diracs[lcs_output_diracs != 0] - lcs_offset
x_values_for_interp = x_values[lcs_output_diracs != 0]

# create a function that resembles the interpolated array
f = interp1d(x_values_for_interp, lcs_output_for_interp, kind='linear')

# create the interpolated array in the interval of interest
x_values_for_interp = x_values[mse_calc_start_index:mse_calc_end_index]
lcs_output_interpolated = f(x_values_for_interp)

plt_idx_start = int(samplerate*plot_start_s) - mse_calc_start_index
plt_idx_end = int(samplerate*plot_end_s) - mse_calc_start_index

#ax[3].figure('Reconstruction with linear interpolation')
ax[3].plot(x_plot, data_deinterleaved_normalized[0][int(samplerate*plot_start_s) : int(samplerate*plot_end_s)], label="orig signal")

# the lcs appears to have an offset. The MSE can be lowered a lot, when we add this simple offset to the interpolated version of the signal. 
# we use a standard optimizer function from scipy to find the minimum MSE
calc_mse = lambda offs : (((lcs_output_interpolated + offs - scope_data[0][mse_calc_start_index:mse_calc_end_index])**2)).mean()
optimized_offset = minimize(calc_mse, -0.25, method = 'Nelder-Mead').x
mse_vec = ((lcs_output_interpolated + optimized_offset - scope_data[0][mse_calc_start_index:mse_calc_end_index])**2)

ax[3].plot(x_values_for_interp/samplerate, lcs_output_interpolated, label='linear interpolation')
ax[3].plot(x_values_for_interp/samplerate, lcs_output_interpolated+optimized_offset, label='interp with opt. offset: ' + str(optimized_offset))
ax[3].plot(x_values_for_interp/samplerate, mse_vec, label='mse: {:.3f}'.format(mse_vec.mean()) )

#ax[3].legend(loc='lower left')
ax[3].set_ylabel('amplitude')
ax[3].set_ylim([-1,2])
ax[3].set_title('Reconstruction with linear interpolation')
ax[3].legend(loc='lower left')

