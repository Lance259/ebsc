#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:53:59 2023

@author: lhinz
"""

import numpy as np
import matplotlib.pyplot as plt


def x(u, a=1, b=1):
    return np.cos(2*u) + 2*np.cos(-u) + a*np.cos(u/2) + b*np.cos(-u/2)
    
def y(u, a=1, b=1):
    return np.sin(2*u) + 2*np.sin(-u) + a*np.sin(u/2) + b*np.sin(-u/2)

a = 5
b = 10

us = np.linspace(0, 4*np.pi, 100)
xs = [x(u,a,b) for u in us]
ys = [y(u,a,b) for u in us]
plt.plot(xs, ys)