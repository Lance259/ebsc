#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:49:13 2024

@author: lhinz
"""
import numpy as np
import matplotlib.pyplot as plt

v_out = [2.15, 
        2.1,
        2.06,
        2.03,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02,
        2.01,
        2,
        1.98,
        1.89,
        1.75,
        1.6,
        1.22]

v_in= [2.16,
        2.09,
        2.05,
        2.03,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02,
        2.02]

freq = [5000,
2000,
1000,
500,
200,
100,
50,
30,
20,
10,
5,
3,
2,
1,
0.8,
0.5,
0.3
]

phase=[0,
0,
0,
0,
0,
0,
0,
-0.8,
-1.3,
-2.4,
-4.8,
-8,
-11.9,
-22.8,
-27.7,
-41.4,
-59
]
v_out = np.array(v_out)
v_in = np.array(v_in)

A = 20*np.log10(v_out/v_in)


fig, ax1 = plt.subplots()

lns1 = ax1.semilogx(freq, A, label = 'Gain', color='r')
ax1.tick_params(axis='y', labelcolor='r')
ax2 = ax1.twinx()
lns2 = ax2.semilogx(freq, phase, label='Phase', color='b')
ax2.tick_params(axis='y', labelcolor='b')


ax1.set_xlabel("frequency / Hz")
ax1.set_ylabel("20log(v_out / v_in) / dB", color = 'r')
ax2.set_ylabel("Phase / deg", color = 'b')

# added these three lines
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc=0)

plt.title("Bode Plot Pre-Amplifier")
fig.tight_layout()


