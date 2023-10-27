'''
Created on 2022-10-08 11:44:13
@author: caitgrasso

Description: Computes distances between each pair of cells' center of masses and reads out as a matrix.

'''

import os
import pandas as pd
import numpy as np
import math
from glob import glob

in_dir = 'spatial_data/raw_spatial'

for scan in glob(in_dir+ '/bot*centroids_pruned.csv'):

    print(scan)
    
    com_df = pd.read_csv(scan, header=0)

    # Compute distances between each pair of cells

    N0 = len(com_df) # number of cells

    distance_mat = np.zeros(shape=(N0,N0))

    for i in range(N0): # Every unique pair of cells
        for j in range(i):
            p = (com_df.loc[i]["x"], com_df.loc[i]["y"])
            q = (com_df.loc[j]["x"], com_df.loc[j]["y"])

            dist = np.abs(math.dist(p,q)) # euclidean distance
            
            # set edge weight to the inverse of distance (closer nodes have a larger edge weight)
            # was set to 1/distance
            distance_mat[i][j] = dist # Make sure to add both upper and lower triangles
            distance_mat[j][i] = dist

    # print(distance_mat)
    np.savetxt('spatial_data/distance_matrices/distmat_pruned_'+scan.split('/')[-1], distance_mat, delimiter=",")


