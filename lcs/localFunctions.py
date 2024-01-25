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
    finds the index of array a where value b first occurs
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
    
    
def calc_sample_mse(orig_signal, sample_vec, skip_first_n_samples=10, no_of_samples_for_mse=100):

    orig_signal_samples_only = orig_signal[sample_vec != 0]
    sample_vec_samples_only = sample_vec[sample_vec != 0]
    
    try:
        mse_vec = ((orig_signal_samples_only[skip_first_n_samples:skip_first_n_samples+no_of_samples_for_mse] - sample_vec_samples_only[skip_first_n_samples:skip_first_n_samples+no_of_samples_for_mse])**2)
    except Exception:
        print("number of samples is only " + len(orig_signal_samples_only) + ", but " + len(no_of_samples_for_mse+skip_first_n_samples) + " samples were requested to calc mse. (including skip samples)")
        mse_vec = ((orig_signal_samples_only[skip_first_n_samples:] - sample_vec_samples_only[skip_first_n_samples:])**2)
    
    sample_mse = mse_vec.mean()
    return sample_mse

def calc_rmtd_lmtd(orig_signal, scaled_diracs, t0=500, y_tol=0.005, us_factor=4):
    #t0 = 500 #interval around the dirac of interest
    sample_events = np.arange(0,len(scaled_diracs),1)[scaled_diracs != 0]
    
    rmtd = 0
    lmtd = 0
    rmtd_vec = []
    lmtd_vec = []
    num_rightshifts = 0
    num_leftshifts = 0
    num_no_shift_detectable = 0
    for t_k in sample_events:
        s_hat_k = scaled_diracs[t_k]
        
        start_idx = t_k-t0
        end_idx = t_k+t0+1
        
        search_signal = orig_signal[start_idx:end_idx]
        #search_signal must be upsampled to find s_hat_k with a small y_tol
        assert (2*t0 + 1 == len(search_signal))
        search_signal_us = resample(search_signal, (2*t0 + 1)*us_factor)
        find_idx = find_first_occurrence_index(search_signal_us, s_hat_k, y_tol)
        
        if(find_idx is None):
            num_no_shift_detectable += 1
            find_idx = t0
            
        find_idx = us_factor*t0 - find_idx #find_idx > 0 => right shift, find_idx < 0 => left shift
        
        if(find_idx > 0): # right shift detected
            rmtd += find_idx/us_factor
            rmtd_vec.append(find_idx/us_factor)
            num_rightshifts += 1
        else: # left shift detected
            lmtd += find_idx/us_factor
            lmtd_vec.append(find_idx/us_factor)
            num_leftshifts += 1
        
    rmtd /= (num_rightshifts if num_rightshifts > 0 else 1)
    lmtd /= (num_leftshifts if num_leftshifts > 0 else 1)
    
    assert (rmtd == np.array(rmtd_vec).mean())
    assert (lmtd == np.array(lmtd_vec).mean())
    
    
    return rmtd, lmtd, num_no_shift_detectable