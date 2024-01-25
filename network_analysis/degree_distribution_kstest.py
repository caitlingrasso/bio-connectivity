'''
Created on 2024-01-20 11:03:04
@author: caitgrasso
'''

import numpy as np
from scipy import stats
import pandas as pd

BOTS = ['bot_01', 'bot_02', 'bot_03', 'bot_04', 'bot_05', 'bot_06']

results_df = pd.DataFrame(columns=['bot', 'stat', 'p', 'p-corr'])

for i,BOT in enumerate(BOTS):
    
    # Load degree distribution and null distribution
    degdist_before = np.loadtxt(f'network_analysis_data/degree_distributions/{BOT}_before_mimat_w_degree_distribution.csv', delimiter=',')
    degdist_after = np.loadtxt(f'network_analysis_data/degree_distributions/{BOT}_after_mimat_w_degree_distribution.csv', delimiter=',')

    ksstat, p = stats.ks_2samp(data1=degdist_before, data2=degdist_after) # compares the shapes of the two distributions so it shouldn't matter that they have different ranges (i.e. # nodes in the networks)

    row = {'bot':BOT, 'stat': ksstat, 'p':p, 'p-corr':p*6}

    results_df = results_df.append(row, ignore_index=True)


results_df.to_csv('results/degree_distribution_kstests.csv', index=False)