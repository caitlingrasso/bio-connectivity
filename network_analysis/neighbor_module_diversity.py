import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

def neighbor_diversity(FC, consensus_partition):
    
    diversities = []

    all_fcs = FC[np.triu_indices(FC.shape[0], k=1)]
    num_edges = np.sum(all_fcs!=0)
    num_nodes = FC.shape[0]
    density = (2*num_edges)/(num_nodes*(num_nodes-1))
    num_modules = len(np.unique(consensus_partition))

    for cell in range(FC.shape[0]): # iterate through rows of cells
        indices = np.where(FC[cell] > 0) # ids of neighbor cell
        neighbor_modules = consensus_partition[indices]
        node_diversity = len(np.unique(neighbor_modules))

        diversities.append(node_diversity/num_modules) # normalize by the number of modules
    
    return diversities

BOTS = ['bot_01', 'bot_02', 'bot_03', 'bot_04', 'bot_05', 'bot_06']

all_diversities1 = []
all_diversities2 = []

for BOT in BOTS:

    # read in functional connectivity matrices
    FC1 = np.loadtxt(f'network_inference_data/fc_matrices/{BOT}_before_mimat_w.csv', delimiter=',')
    FC2 = np.loadtxt(f'network_inference_data/fc_matrices/{BOT}_after_mimat_w.csv', delimiter=',')
    N1 = FC1.shape[0]
    N2 = FC2.shape[1]

    # read in consensus partitions
    consensus_partition1 = np.loadtxt(f'network_analysis_data/community_detection/{BOT}_before_mimat_w_consensus_partition.csv', delimiter=',')
    consensus_partition2 = np.loadtxt(f'network_analysis_data/community_detection/{BOT}_after_mimat_w_consensus_partition.csv', delimiter=',')

    diversities1 = neighbor_diversity(FC1, consensus_partition1)
    diversities2 = neighbor_diversity(FC2, consensus_partition2)

    all_diversities1 += diversities1
    all_diversities2 += diversities2

stat, p = mannwhitneyu(all_diversities1, all_diversities2)
print(np.mean(all_diversities1), np.mean(all_diversities2))
print(stat,p)

fig, ax = plt.subplots(1,1, figsize=(3,3),layout='constrained')

before_x = [1]
before_plot = plt.boxplot(all_diversities1,
                            positions=before_x, widths=0.2, patch_artist=True)

after_x = [2]
after_plot = plt.boxplot(all_diversities2,
                            positions=after_x, widths=0.2, patch_artist=True)

for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(before_plot[item], color='dimgray') 
plt.setp(before_plot["boxes"], facecolor = 'lightgray')
plt.setp(before_plot["fliers"], markeredgecolor='dimgray')

for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(after_plot[item], color='darkred')
plt.setp(after_plot["boxes"], facecolor = 'indianred')
plt.setp(after_plot["fliers"], markeredgecolor='darkred')

# Customize labels and title
plt.xticks([1, 2], ['Pre', 'Post'], fontweight='bold', fontsize=15)
ax.tick_params(axis='y', which='major', labelsize=12)
ax.set_ylabel('Frac. neighboring modules', fontweight='bold', fontsize=12)

# h = np.max([np.max(all_diversities1), np.max(all_diversities2)])+1
h = np.max([np.max(all_diversities1), np.max(all_diversities2)])+.02

ax.plot((1,2),(h, h),'k')
if p<0.001:
    text = 'p<0.001'  
elif p<0.01:
    text = 'p<0.01' 
elif p<0.05:
    text = 'p<0.05' 
else:
    text = 'NS'

# plt.text(1.5, h+0.5, text, style='italic', ha='center', fontsize=12)
plt.text(1.5, h+0.01, text, style='italic', ha='center', fontsize=12)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')

# plt.show()
plt.savefig('../../Desktop/neighbor_diversity.png', dpi=300, bbox_inches='tight')