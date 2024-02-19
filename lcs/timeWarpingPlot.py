#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 11:28:20 2024

@author: lhinz
"""

import numpy as np
import matplotlib.pyplot as plt


stepsize = 0.001
t_max = 10
plot_periods = 5
f0 = 1/(t_max/plot_periods)
omega0 = 2*np.pi*f0
samplerate = 2.5*f0

tau = np.arange(0.5,t_max,stepsize)

def h(tau):
    return np.sin(omega0*tau)

def tau_sample(T):
    return T*int(1/(samplerate*stepsize))

def gamma(t):
    return np.sqrt(3*t)

def gamma_dash(t):
    return 3 / (2*np.sqrt(3*t))

def gamma_inv(tau):
    return tau**2  / 3; 


tau_sample = tau[0::int(1/(samplerate*stepsize))]
h_sample = h(tau)[0::int(1/(samplerate*stepsize))]
fig, ax = plt.subplots(4,1,tight_layout=True)
ax[1].plot(tau,h(tau), label='h')
ax[1].stem(tau_sample,h_sample, '+',label='hsample')
ax[1].set_xlabel('$\\tau$')
ax[1].set_ylabel('$h(\\tau)$')
ax[1].set_title('(2) Uniformly sampled $h(\\tau)$')

f = h(gamma(tau))
t = gamma_inv(tau)
t_sample = gamma_inv(tau_sample)
f_sample = h(gamma(tau_sample))
ax[0].plot(t, f)
ax[0].stem(t_sample, f_sample, '+')
ax[0].set_xlabel('$t$')
ax[0].set_ylabel('$f(t)$')
ax[0].set_title('(1) Nonuniformly sampled $f(t)$')

ax[2].plot(t,tau)
ax[2].set_xlabel('$t$')
ax[2].set_ylabel('$\\tau = \\gamma(t)$')
ax[2].set_title('(3) Warping function $\\tau = \\gamma(t)$')

ax[3].plot(t,gamma_dash(t))
ax[3].set_xlabel('$t$')
ax[3].set_ylabel('$d\\gamma(t) / dt$')
ax[3].set_title('(4) Instantaneous sampling rate: derivative of $\\gamma(t)$')


