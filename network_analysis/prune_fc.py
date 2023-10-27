'''
Created on 2023-10-25 12:59:06
@author: caitgrasso
'''

import numpy as np
import pandas as pd

def prune_data(fc_mat, pixels, centroids, series, gsr_whitened_series):

    # Prune FC matrix
    pruned_fc_mat = np.copy(fc_mat)
    pruned_fc_mat = pruned_fc_mat[~np.all(fc_mat == 0, axis=1)]
    pruned_fc_mat = pruned_fc_mat[:,~np.all(fc_mat == 0, axis=0)]

    inds_to_be_removed = np.where(np.all(fc_mat == 0, axis=1))

    # Prune spatial data
    # remove rows in data frames on unconnected nodes
    pixels = pixels.set_index("series_index")
    centroids = centroids.set_index("series_index")
    for ind in inds_to_be_removed:
        pixels.drop(ind, inplace=True)
        centroids.drop(ind, inplace=True)
    
    # Reset the 'series_index' rows to match the pruned FC
    centroids = centroids.reset_index()
    pixels = pixels.reset_index()

    new_series_indices = np.arange(len(centroids))
    curr_series_indices = np.asarray(list(centroids["series_index"]),dtype='int')
    index_map = dict(zip(curr_series_indices, new_series_indices))
    

    centroids['series_index'] = centroids['series_index'].map(index_map)
    pixels['series_index'] = pixels['series_index'].map(index_map)


    # Prune time series
    series = series[~np.all(fc_mat == 0, axis=0)]
    gsr_whitened_series = gsr_whitened_series[~np.all(fc_mat == 0, axis=0)]

    # Check dimensions

    print(fc_mat.shape)
    print(pruned_fc_mat.shape)
    print(series.shape[0])
    print(gsr_whitened_series.shape[0])
    print(len(centroids))
    print(len(np.unique(list(pixels['series_index']))))

    return pruned_fc_mat, pixels, centroids, series, gsr_whitened_series


BOTS = ['bot_01', 'bot_02', 'bot_03', 'bot_04', 'bot_05', 'bot_06']

for BOT in BOTS:

    # Load data
    FC1 = np.loadtxt(f'network_inference_data/fc_matrices/{BOT}_before_mimat_w.csv', delimiter=',')
    FC2 = np.loadtxt(f'network_inference_data/fc_matrices/{BOT}_after_mimat_w.csv', delimiter=',')

    pixels1 = pd.read_csv(f'spatial_data/raw_spatial/{BOT}_before_pixels.csv')
    pixels2 = pd.read_csv(f'spatial_data/raw_spatial/{BOT}_after_pixels.csv')

    centroids1 = pd.read_csv(f'spatial_data/raw_spatial/{BOT}_before_centroids.csv')
    centroids2 = pd.read_csv(f'spatial_data/raw_spatial/{BOT}_after_centroids.csv')

    series1 = np.loadtxt(f'network_inference_data/series_raw/{BOT}_before_series.csv', delimiter=',')
    series2 = np.loadtxt(f'network_inference_data/series_raw/{BOT}_after_series.csv', delimiter=',')

    gsr_whitened_series1 = np.loadtxt(f'network_inference_data/series_gsr_whitened/{BOT}_before_entrate.csv', delimiter=',')
    gsr_whitened_series2 = np.loadtxt(f'network_inference_data/series_gsr_whitened/{BOT}_after_entrate.csv', delimiter=',')

    # Prune
    pruned_FC1, pixels_pruned1, centroids_pruned1, series_pruned1, gsr_whitened_series_pruned1 = prune_data(FC1, pixels1, centroids1, series1, gsr_whitened_series1)
    pruned_FC2, pixels_pruned2, centroids_pruned2, series_pruned2, gsr_whitened_series_pruned2 = prune_data(FC2, pixels2, centroids2, series2, gsr_whitened_series2)

    # Save data
    np.savetxt(f'network_inference_data/fc_matrices/{BOT}_before_mimat_w_pruned.csv', pruned_FC1, delimiter=',')
    np.savetxt(f'network_inference_data/fc_matrices/{BOT}_after_mimat_w_pruned.csv', pruned_FC2, delimiter=',')

    pixels_pruned1.to_csv(f'spatial_data/raw_spatial/{BOT}_before_pixels_pruned.csv')
    pixels_pruned2.to_csv(f'spatial_data/raw_spatial/{BOT}_after_pixels_pruned.csv')

    centroids_pruned1.to_csv(f'spatial_data/raw_spatial/{BOT}_before_centroids_pruned.csv')
    centroids_pruned2.to_csv(f'spatial_data/raw_spatial/{BOT}_after_centroids_pruned.csv')

    np.savetxt(f'network_inference_data/series_raw/{BOT}_before_series_pruned.csv', series_pruned1, delimiter=',')
    np.savetxt(f'network_inference_data/series_raw/{BOT}_after_series_pruned.csv', series_pruned2, delimiter=',')

    np.savetxt(f'network_inference_data/series_gsr_whitened/{BOT}_before_entrate_pruned.csv', gsr_whitened_series_pruned1, delimiter=',')
    np.savetxt(f'network_inference_data/series_gsr_whitened/{BOT}_after_entrate_pruned.csv', gsr_whitened_series_pruned2, delimiter=',')

