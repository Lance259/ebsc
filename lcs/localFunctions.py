#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:22:05 2024

@author: lhinz
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import write
from scipy.interpolate import interp1d
from scipy.signal import resample 

import pandas as pd
import seaborn as sns

def deinterleave(data, num_streams, block_size):
    """
    Takes array in format [A A A B B B C C C A A A B B B C C C ...]
    and converts it to [[A A A A A...]
                        [B B B B B ...]
                        [C C C C C ...]]

    Parameters
    ----------
    data : np array
        interleaved data array from gnuRadio
    num_streams : int
        number of data streams (in example above: 3)
    block_size : int
        blocks of coherent data from one stream, before the next stream starts (in example above: 3)

    Returns
    -------
    output : np array
        deinterleaved array of shape (num_streams, len(data)/num_streams)

    """
    blocks = np.reshape(data, (-1, block_size))
    print(blocks.shape)
    output = np.zeros((num_streams, int(len(data)/num_streams)))
    
    for i in range(num_streams):
        output[i,:] = blocks[i::num_streams].flatten()
        
    return output


def gradient_to_dirac_scaled(grad_signal, grad_thresh, amp):
    """
    

    Parameters
    ----------
    grad_signal : 1D np array
        gradient peak vector
    grad_thresh : float
        threshold signal for gradient peaks
    amp : float
        amplitude for output diracs

    Returns
    -------
    output: 1D np array
        array with dirac peaks (zero otherwise) with height = amplitude, at times where the input gradient peaks exceeded threshold

    """
    #edge_detect_kernel = np.array([-1, 2, -1])
    #output = np.zeros(len(grad_signal))
    edges = np.convolve((np.abs(grad_signal) > grad_thresh).astype(int), np.array([1,-1]), mode="same") 
    return (edges > 0).astype(int)*amp

    
def gradient_to_dirac(grad_signal, grad_thresh, amp):
    """
    Parameters
    ----------
    grad_signal : numpy array
        input array, typically a gradient of a digital signal
    grad_thresh : float
        threshold for the peaks in the gradient signal to exceed and create a dirac at that position
    amp : float
        height of the output dirac .

    Returns
    -------
    numpy array
        an array with the same length as grad_signal. Is zero everywhere, except
        for the indices where the gradient signal first crossed the threshold.
        The output array is +1 where a positive gradient was detected (rising edge)
        and it is -1 where a negative gradient was detected (falling edge). 
        
        This is important for decoding of the state

    """
    edges = np.convolve((np.abs(grad_signal) > grad_thresh).astype(int), np.array([1,-1]), mode="same") 
    
    pos_edges = ((edges > 0).astype(int)*grad_signal > 0).astype(int)
    neg_edges = -((edges > 0).astype(int)*grad_signal < 0).astype(int)    
    return pos_edges + neg_edges

def support(vec):
    """

    Parameters
    ----------
    vec : np array
        any array

    Returns
    -------
    support
        returns the number of places where the input vec is non-zero (its support)

    """
    return (vec != 0).astype(int).sum()


def find_first_occurrence_index(a, b, eps=0):
    """
    finds the index of array a where value b first occurs, 
    within a tolerance range of [b-eps, b+eps]
    """
    # Convert the input list to a numpy array
    a = np.array(a)
    
    # Find the indices where the value is equal to b
    indices = np.where((a >= b-eps) & (a <= b+eps))[0]
    
    # Return the index of the first occurrence
    if len(indices) > 0:
        return indices[0]
    else:
        # Handle the case where b is not in the array
        return None
   
def extract_single_state_transition(arr, a):
    """
    Accepts a numpy array `arr` of any length and a number `a`.
    Returns an array of the same length as `arr`, filled with zeros
    except where the entry in `arr` matches the number `a`.
    """
    # Create an array of zeros with the same shape as arr
    result = np.zeros_like(arr)
    
    # Find indices where arr equals a
    indices = np.where(arr == a)
    
    # Fill result array with a at the corresponding indices
    result[indices] = a
    
    return result
    
def state_transition(in_state, in_signals):
    if(in_state == 0):
        if(np.array_equal(in_signals, [0,0,1])): # state trans 0->1
            return 1, 1
    elif(in_state == 1):
        if(np.array_equal(in_signals, [0,1,0])): # # state trans 1->2
            return 2, 2
        elif(np.array_equal(in_signals, [0,0,-1])): # state trans 1->0
            return 0, 1
    elif(in_state == 2):
        if(np.array_equal(in_signals, [0,0,-1])): # state trans 2->3
            return 3, 3
        elif(np.array_equal(in_signals, [0,-1,0])): # state trans 2->1
            return 1, 2
    elif(in_state == 3):
        if(np.array_equal(in_signals, [1,0,0])): # state trans 3->4
            return 4, 4
        elif(np.array_equal(in_signals, [0,0,1])): # state trans 3->2
            return 2, 3
    elif(in_state == 4):
        if(np.array_equal(in_signals, [0,0,1])): #  state trans 4->5
            return 5, 5
        elif(np.array_equal(in_signals, [-1,0,0])): # state trans 4->3
            return 3, 4
    elif(in_state == 5):
        if(np.array_equal(in_signals, [0,-1,0])): # state trans 5->6
            return 6, 6
        elif(np.array_equal(in_signals, [0,0,-1])): # state trans 5->4
            return 4, 5
    elif(in_state == 6):
        if(np.array_equal(in_signals, [0,1,0])): # state trans 6->5
            return 5, 6
    else:
        return -1
    return in_state, 0 # stay in same state, when no matching state transition (means that the assumed start state was wrong)


def state_to_lcs (state_array, level_list):
    return np.array(level_list)[state_array]

    
def calc_sample_mse(orig_signal, scaled_diracs, skip_first_n_samples=10, no_of_samples_for_mse=100):
    """

    Parameters
    ----------
    orig_signal : 1D np array
        original reference vector
    scaled_diracs : 1D np array
        vector of the same length as orig_signal, containing only scaled diracs at the sample times
    skip_first_n_samples : int, optional
        In the beginning of the measurement, some samples might be noisy. This parameter is used to skip some sample instances for the mse calculation. The default is 10.
    no_of_samples_for_mse : int, optional
        Number of sample instances which are used to calculate the mse. The default is 100.

    Returns
    -------
    sample_mse : float32
        The mean squared error / amplitude deviation for the number of considered samples.

    """
    orig_signal_samples_only = orig_signal[scaled_diracs != 0]
    scaled_diracs_samples_only = scaled_diracs[scaled_diracs != 0]
    
    try:
        mse_vec = ((orig_signal_samples_only - scaled_diracs_samples_only)**2) #((orig_signal_samples_only[skip_first_n_samples:skip_first_n_samples+no_of_samples_for_mse] - scaled_diracs_samples_only[skip_first_n_samples:skip_first_n_samples+no_of_samples_for_mse])**2)
    except Exception:
        pass
        #print("number of samples is only " + len(orig_signal_samples_only) + ", but " + len(no_of_samples_for_mse+skip_first_n_samples) + " samples were requested to calc mse. (including skip samples)")
        #mse_vec = ((orig_signal_samples_only[skip_first_n_samples:] - scaled_diracs_samples_only[skip_first_n_samples:])**2)
    
    sample_mse = mse_vec.mean()
    return sample_mse

def calc_rmtd_lmtd(orig_signal, scaled_diracs, t0=500, y_tol=0.005, us_factor=4, returnArray=True):
    """

    Parameters
    ----------
    orig_signal : 1D np array
        original reference vector
    scaled_diracs : 1D np array
        vector of the same length as orig_signal, containing only scaled diracs at the sample times
    t0 : int, optional
        The time window in which to search the original signal for the sample value. The default is 500.
    y_tol : float, optional
        Allowed deviation between sample value and value in original signal. The default is 0.005.
    us_factor : int, optional
        Upsampling and interpolation of the original signal allows for smaller y_tol and more accurate results. The default is 4.
    returnArray : bool, optional
        return either vectors of all occurences of right/left time deviation (True) or return mean rmtd/lmtd and standard deviation directly . The default is False.

    Returns
    -------
    1d_array, 1d_array, int
        returns vectors of right time deviations, left time deviations 
        and a number of not detectable deviations (the sample value
        did not correspond to any value of the original signal, 
        within the search interval t0 and amplitude tolerance y_tol)

    """
    #t0 = 500 #interval around the dirac of interest
    sample_events = np.arange(0,len(scaled_diracs),1)[scaled_diracs != 0]
    rmtd_vec = []
    lmtd_vec = []

    num_no_shift_detectable = 0
    for t_k in sample_events:
        s_hat_k = scaled_diracs[t_k]
        
        start_idx = t_k-t0
        end_idx = t_k+t0+1
        
        search_signal = orig_signal[start_idx:end_idx]
        
        if(len(search_signal) > 0):
            #search_signal must be upsampled to find s_hat_k with a small y_tol

            search_signal_us = resample(search_signal, (len(search_signal))*us_factor)
            find_idx = find_first_occurrence_index(search_signal_us, s_hat_k, y_tol)
            
            if(find_idx is None):
                num_no_shift_detectable += 1
                find_idx = us_factor*t0
                
            find_idx = us_factor*t0 - find_idx #find_idx > 0 => right shift, find_idx < 0 => left shift
            
            if(find_idx > 0): # right shift detected
                rmtd_vec.append(find_idx/us_factor)
            else: # left shift detected
                lmtd_vec.append(find_idx/us_factor)
    
    perc_no_shift_detectable = 1
    if(len(sample_events) != 0):
        perc_no_shift_detectable = num_no_shift_detectable/(len(sample_events))

    if(returnArray):
        return np.array(rmtd_vec), np.array(lmtd_vec), perc_no_shift_detectable
    else:
        return (np.array(rmtd_vec).mean(), np.array(rmtd_vec).std()), (np.array(lmtd_vec).mean(), np.array(lmtd_vec).std()), perc_no_shift_detectable
    
    
def calc_error_metrics(orig_signal, scaled_diracs, samplerate, freq_of_meausred_signal, plot_mtd_details=True, titlestring='plottitle'):
    """
    

    Parameters
    ----------
    orig_signal : 1D np array length N
        array containing the original signal with respect to which to calc the metrics. 
    scaled_diracs : np array length LxN
        matrix with N columns, and L rows. each row contains the event-based samples for one level. A sample is given as a correctly scaled single non-zero entry in an array that is zero otherwise. 
    samplerate : int
        sample rate for correct display of time unit 
    freq_of_meausred_signal : float
        used to determine the t_0 interval for MTD calculation
    plot_mtd_details : bool, optional
        Whether to plot LMTD/RMTD for each level. The default is True.
    titlestring : string, optional
        Used in plot titles. The default is 'plottitle'.

    Returns
    -------
    error_metrics_average : np array of length 6
        contains: [mse, rmtd_mean, rmtd_std, lmtd_mean, lmtd_std, perc_no_undetectable]
    error_metrics_each_level : TYPE
        contains: [mse, rmtd_mean, rmtd_std, lmtd_mean, lmtd_std, perc_no_undetectable]

    """
    ############## calculate error metrics ###############
    # calculate the mse at sample - instances
    samples_per_period = int(samplerate / freq_of_meausred_signal)
    sample_mse = []
    rmtd_mean_arr = []
    lmtd_mean_arr= []
    no_det_arr = []
    time_deviations_list = []

    for i in range(scaled_diracs.shape[0]):
        sample_mse.append(calc_sample_mse(orig_signal, scaled_diracs[i])) #(data_deinterleaved_normalized[0], level_crossings_scaled[i], skip_first_n_samples, no_of_samples_for_mse))
        rmtd, lmtd, not_detectable = calc_rmtd_lmtd(orig_signal, scaled_diracs[i], t0=int(samples_per_period/8), y_tol=0.005, us_factor=16, returnArray=True)

        rmtd *= 1e3
        lmtd *= 1e3
        
        for r in rmtd:
            time_deviations_list.append(dict([('time deviation / ms', r/samplerate), ('direction', 'rmtd'), ('level', i+1)]))
        for l in lmtd:
            time_deviations_list.append(dict([('time deviation / ms', l/samplerate), ('direction', 'lmtd'), ('level', i+1)]))
            
        rmtd_mean_arr.append((rmtd.mean(), rmtd.std()))
        lmtd_mean_arr.append((lmtd.mean(), lmtd.std()))
        no_det_arr.append(not_detectable)
        
    error_metrics_each_level = np.nan_to_num(np.array([sample_mse, np.array(rmtd_mean_arr)[:,0]/samplerate,np.array(rmtd_mean_arr)[:,1]/samplerate, np.array(lmtd_mean_arr)[:,0]/samplerate ,np.array(lmtd_mean_arr)[:,1]/samplerate, no_det_arr]))
    # reference error metrics: [mse, rmtd_mean, rmtd_std, lmtd_mean, lmtd_std, perc_no_undetectable]

    error_metrics_average = [q.mean(axis=0) for q in error_metrics_each_level]

    if(plot_mtd_details):
        time_deviations = pd.DataFrame(time_deviations_list, columns = ['time deviation / ms', 'direction', 'level'])
        plt.figure('lmtd_rmtd plot')
        plt.title('Mean Time Deviation, ' + titlestring)
        sns.barplot(time_deviations, x="time deviation / ms", y="level", hue="direction", orient='h')
    
    return error_metrics_average, error_metrics_each_level

