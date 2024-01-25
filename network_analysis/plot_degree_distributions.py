'''
Created on 2023-10-25 12:51:32
@author: caitgrasso
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from collections import Counter
from scipy.stats import linregress
import os
from matplotlib.patches import Patch
import matplotlib as mpl
from matplotlib.colors import ListedColormap

plt.rcParams.update({'font.size': 10, 'font.family': 'arial'})

fig,axes = plt.subplots(3, 4, figsize=(11,7))

axes_lst = axes.ravel()

BOTS = ['bot_01', 'bot_02', 'bot_03', 'bot_04', 'bot_05', 'bot_06']

# Get axes range for each pair (before, after) of degree distributions
maxvals = []
for i,BOT in enumerate(BOTS):

    # Load degree distribution and null distribution
    degdist1 = np.loadtxt(f'network_analysis_data/degree_distributions/{BOT}_before_mimat_w_degree_distribution.csv', delimiter=',')
    degdist2 = np.loadtxt(f'network_analysis_data/degree_distributions/{BOT}_after_mimat_w_degree_distribution.csv', delimiter=',')
    
    maxvals.append(np.max((np.max(degdist1), np.max(degdist2))))

ax_cntr = 0

for i,BOT in enumerate(BOTS):

    for phase in ['before', 'after']:
    
        ax = axes_lst[ax_cntr] 

        # Load degree distribution and null distribution
        degdist = np.loadtxt(f'network_analysis_data/degree_distributions/{BOT}_{phase}_mimat_w_degree_distribution.csv', delimiter=',')
        null_degdist = np.loadtxt(f'network_analysis_data/null_degree_distributions/{BOT}_{phase}.csv', delimiter=',')

        ax.scatter(np.arange(1, len(degdist)+1), sorted(degdist), s=30, c='dimgray', edgecolors='k', alpha=0.5)
        ax.scatter(np.arange(1, len(null_degdist)+1), sorted(null_degdist), s=30, c='darkred', alpha=0.5, marker='D')


        ax.set_ylim([0,maxvals[i]])
    
        # if ax_cntr<4:
        #     ax.set_title(phase,fontsize=15,fontweight='bold')
        if ax_cntr%4==0:
            ax.set_ylabel('degree', fontweight='bold', fontsize=15)
        if ax_cntr>7:
            ax.set_xlabel('rank', fontweight='bold', fontsize=15)
        if ax_cntr==0:
            ax.legend(['empirical', 'null'], frameon=False, fontsize=12)

        p = 'pre' if phase=='before' else 'post'

        ax.annotate(f'O{i+1}-{p}', xy=(1, 0),  # Coordinates for the bottom right corner
             xycoords='axes fraction',  # Use axes fraction for specifying coordinates
             xytext=(-5, 5),  # Offset of the text from the specified coordinates
             textcoords='offset points',  # Use offset points for specifying text offset
             ha='right',  # Horizontal alignment
             va='bottom',  # Vertical alignment
             fontsize=10,  # Adjust the font size as needed
             fontweight='bold',
             style='italic',
             bbox=dict(boxstyle='square,pad=0.2', edgecolor='gainsboro', facecolor='gainsboro'))

        ax_cntr+=1

# plt.show()
plt.savefig(f'../../Desktop/degree_distributions.png', dpi=500, bbox_inches='tight')