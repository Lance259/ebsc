#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:44:06 2023

@author: lhinz
"""

import pyvisa as visa
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy

############ USER PARAMS ############
samples = 1<<15
period_s = 7
csv_input_file = '/home/lhinz/Seafile/MasterS3/MasterProject/ebsc/lcs/ekgData/survey_data_7s/N_100_2225_4744.csv'#'/home/lhinz/Seafile/MasterS3/MasterProject/ebsc/lcs/ekgData/survey_data_7s/N_100_2225_4744.csv'
connect_device = True


########## AWG Interface INIT #############
if(connect_device):
    rm = visa.ResourceManager()
    rm.list_resources()
    inst = rm.open_resource('TCPIP::169.254.2.20::INSTR')
    
    inst.read_termination = '\n'
    inst.write_termination = '\n'
    print(inst.query('*IDN?'))


# generate an arbitrary wave
"""
t = np.arange(0, samples-1, 1)
wave = np.sinc(4.5*2*(t-samples/2)/samples)
"""

ekg_data = np.genfromtxt(csv_input_file, delimiter=",")
mlii_data = ekg_data[:,0][~np.isnan(ekg_data[:,0])]

mlii_data_resampled = scipy.signal.resample(mlii_data, samples)

"""
t_norm = np.linspace(0, 7, len(mlii_data))
t_resampled = np.linspace(0, 7, len(mlii_data_resampled))
plt.plot(t_norm, mlii_data, t_resampled, mlii_data_resampled)
"""

wave = mlii_data_resampled
wave = wave/np.max(wave)
#plt.plot(wave)

value_str = ''
for value in wave: 
    value_str += ', '  + str(value)


    
    
    
    
############## SCPI COMMANDS ##############
    
inst.write('DATA VOLATILE' + value_str)
inst.write('FUNC:USER VOLATILE')
inst.write('FUNC USER')
inst.write('VOLT:UNIT VPP')
inst.write('VOLT 1.3')
inst.write('FREQ ' + str(1/period_s))

inst.write('OUTP ON')
#inst.write('DATA COPY 0, VOLATILE' + value_str)
