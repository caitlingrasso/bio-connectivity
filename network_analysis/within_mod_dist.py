'''
Created on 2023-09-06 10:47:56
@author: caitgrasso
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

def get_distances(distmat, consensus_partition):

    max_distance = np.max(distmat)

    all_distances = list(distmat[np.triu_indices(distmat.shape[0], k=1)])

    within = []
    outside = []
    # iterate through modules
    for module in np.unique(consensus_partition):
        all_inds = np.arange(len(consensus_partition))
        node_inds_in_module = all_inds[consensus_partition==module]
        node_inds_out_of_module = all_inds[consensus_partition!=module]

        # compute within module distances
        for i in range(len(node_inds_in_module)):
            for j in range(i):
                node1 = node_inds_in_module[i]
                node2 = node_inds_in_module[j]

                within.append(distmat[node1,node2]/max_distance) # normalize by the maximum distance between two nodes in the network?
                # within.append(distmat[node1,node2])
        
        # compute out of module distances
        for i in range(len(node_inds_in_module)):
            for j in range(len(node_inds_out_of_module)):
                node1 = node_inds_in_module[i]
                node2 = node_inds_out_of_module[j]

                outside.append(distmat[node1,node2]/max_distance)
                # outside.append(distmat[node1,node2]) # normalize by the maximum distance between two nodes in the network?

    return within, outside, all_distances

BOTS = ['bot_01', 'bot_02', 'bot_03', 'bot_04', 'bot_05', 'bot_06']

all_withins1 = []
all_withins2 = []
all_outsides1 = []
all_outsides2 = []
all_distances1 = []
all_distances2 = []

for BOT in BOTS:

    # read in functional connectivity matrices
    FC1 = np.loadtxt(f'network_inference_data/fc_matrices/{BOT}_before_mimat_w.csv', delimiter=',')
    FC2 = np.loadtxt(f'network_inference_data/fc_matrices/{BOT}_after_mimat_w.csv', delimiter=',')
    N1 = FC1.shape[0]
    N2 = FC2.shape[1]

    # read in consensus partitions
    consensus_partition1 = np.loadtxt(f'network_analysis_data/community_detection/{BOT}_before_mimat_w_consensus_partition.csv', delimiter=',', dtype=int)
    consensus_partition2 = np.loadtxt(f'network_analysis_data/community_detection/{BOT}_after_mimat_w_consensus_partition.csv', delimiter=',', dtype=int)

    # read in distance matrices
    distances_mat1 = np.asarray(np.loadtxt(f'spatial_data/distance_matrices/distmat_{BOT}_before_centroids.csv', delimiter=','))
    distances_mat2 = np.asarray(np.loadtxt(f'spatial_data/distance_matrices/distmat_{BOT}_after_centroids.csv', delimiter=','))

    within1,outside1,all1 = get_distances(distances_mat1,consensus_partition1)
    within2,outside2,all2 = get_distances(distances_mat2,consensus_partition2)
    
    all_withins1 += within1
    all_withins2 += within2
    all_outsides1 += outside1
    all_outsides2 += outside2
    all_distances1 += all1
    all_distances2 += all2

# Statistical testing
ksstat_within, p_within = stats.ks_2samp(data1=all_withins1, data2=all_withins2)
print('WITHIN:')
print(ksstat_within, p_within)
print()

print('OUTSIDE:')
ksstat_outside, p_outside = stats.ks_2samp(data1=all_outsides1, data2=all_outsides2)
print(ksstat_outside, p_outside)

print('ALL:')
ksstat_all, p_all = stats.ks_2samp(data1=all_distances1, data2=all_distances2)
print(ksstat_all, p_all)

data = {'pre-within': all_withins1,
        'post-within': all_withins2,
        'pre-outside': all_outsides1,
        'post-outside': all_outsides2}
        # 'pre-all': all_distances1,
        # 'post-all': all_distances2}

locations = [1,1.75,3,3.75]
# locations = [1,1.75,3,3.75,5,5.75]

# Set custom edge colors for each group
edge_colors = {'pre-within': 'dimgray',
               'post-within': 'darkred',
                'pre-outside': 'dimgray',
                'post-outside': 'darkred'}
                # 'pre-all': 'dimgray',
                # 'post-all': 'darkred'}

face_colors = {'pre-within': 'lightgray',
               'post-within': 'indianred',
                'pre-outside': 'lightgray',
                'post-outside': 'indianred'}
                # 'pre-all': 'lightgray',
                # 'post-all': 'indianred'}

labels = ['']

# Create violin plots with custom edge colors
fig, ax = plt.subplots(1,1,figsize=(6,3), layout='constrained')
# fig, ax = plt.subplots(1,1,figsize=(7,3), layout='constrained')

labels = []
for i, group in enumerate(data):
    values = data[group]

    parts = ax.violinplot(values, positions=[locations[i]], showmeans=True)
    
    for pc in parts['bodies']:
        pc.set_edgecolor(edge_colors[group])
        pc.set_alpha(0.8)
        pc.set_linewidth(2)
        pc.set_facecolor(face_colors[group])

    parts['cbars'].set_edgecolor(edge_colors[group])
    parts['cmeans'].set_edgecolor(edge_colors[group])
    parts['cmins'].set_edgecolor(edge_colors[group])
    parts['cmaxes'].set_edgecolor(edge_colors[group])
    
    if i<2:
        if 'pre' in group:
            label = 'Pre'
        else:
            label = 'Post'
        labels.append((mpatches.Patch(facecolor=face_colors[group],edgecolor=edge_colors[group]), label))

ax.set_xticks([1.375,3.375], ['Within module', 'Between module'], fontweight='bold', fontsize=15)
# ax.set_xticks([1.375,3.375,5.375], ['Within module', 'Outside of module', 'All'], fontweight='bold', fontsize=15)
ax.tick_params(axis='y', which='major', labelsize=12)
ax.set_ylabel('Normalized distance', fontweight='bold', fontsize=15)

plt.legend(*zip(*labels), loc='upper center')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(True, linestyle='--', axis='y', linewidth=0.5, alpha=0.7, color='gray')

# plt.show()
plt.savefig('../../Desktop/within_module_distance.png', dpi=500, bbox_inches='tight')