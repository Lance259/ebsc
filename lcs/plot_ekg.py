#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:22:05 2023

@author: lhinz
"""
import numpy as np
import matplotlib.pyplot as plt

filename_csv= '/home/lhinz/Seafile/MasterS3/MasterProject/ebsc/lcs/ekgData/survey_data_7s/AFIB_202_411724_414243.csv'

ekg_data = np.genfromtxt(filename_csv, delimiter=",")
mlii_data = ekg_data[:,0][~np.isnan(ekg_data[:,0])]

x_values = np.arange(0, len(mlii_data), 1) / 360
plt.plot(x_values, mlii_data)