#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 13:44:46 2022

@author: thosvarley
"""

import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import zscore, pearsonr, linregress
from copy import deepcopy
from jpype import isJVMStarted, startJVM, getDefaultJVMPath, JPackage


#%% 
# MODULAR FUNCTIONS

def start_jvm(jar_location):
    """
    Getting the java virtual machine running so we can 
    call JIDT functions.

    Parameters
    ----------
    jar_location : str
        The absolute path to wherever the JIDT .jar is.

    Returns
    -------
    None.

    """
    # Add JIDT jar library to the path
    # Start the JVM (add the "-Xmx" option with say 1024M if you get crashes due to not enough memory space)
    if isJVMStarted() == False: # It will choke if you try and start the JVM when it's already running.
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=" + jar_location)


def artifact_filter(series, threshold = 2.0):
    """
    Attempts to fitler artifacts by removing all frames where
    the rate of change is implausibly high based on the dynamics
    of the system.

    Parameters
    ----------
    series : np.ndarray
        The raw time series in channels x time format.
    threshold : float
        The number of standard deviations to consider as the filter threshold. 
        The default is 2.0.

    Returns
    -------
    np.ndarray
        The series object with the offending columns removed. 

    """    
    diff = np.abs(np.diff(series, axis=-1)) # The aboslute value of the derivative.
    lim = threshold * diff.std() # The threshold above which to filter
    
    prop = (diff > lim).sum(axis=0) / diff.shape[0] # Find the proportion of channels at time t above the threshold.
    where = np.where(prop < 0.1)[0] # Keep any frame with 
    reject = np.where(prop > 0.1)[0]
        
    return deepcopy(series[:, where]), reject


def global_signal_regression(series):
    """
    Regresses out the global signal (mean of the time seriese) from 
    each of the channels.

    Parameters
    ----------
    series : np.ndarray
        The raw time series in channels x time format..

    Returns
    -------
    gsr : np.ndarray
        The transformed time series after GSR has been applied.

    """
        
    N0 = series.shape[0] # Number of rows (channels)
    N1 = series.shape[1] # The number of columns (frames)
        
    gsr = np.zeros((N0, N1)) # Initialize GSR array
    mean = series.mean(axis=0) # Compute global signal
            
    for i in range(N0): 
        lr = linregress(mean, series[i]) # Linregress each channel against the GS
        ypred = lr[1] + (lr[0]*mean)
            
        gsr[i,:] = zscore(series[i] - ypred) # Regress out
        
    return gsr


def functional_connectivity(gsr, alpha):
    """
    Computes FC matrices based on linear correlation.

    Parameters
    ----------
    gsr : np.ndarray
        The GSR-d data.
    alpha : float
        Significance threshold.

    Returns
    -------
    corrmat : np.ndarray
        The pearson correlation matrix.
    mimat : np.ndarray
        The mutual information matrix.
    corrmat_correct : np.ndarray
        The pearson corelation marix w/ non-signifcant edges removed.
    mimat_correct : np.ndarray
        The mutual information matrix w/ non-signifcant edges removed. 
    pmat_corr : np.ndarray
        The p-values of each edge.

    """
    N0 = gsr.shape[0] # Number of rows (channels)

    bonferonni = alpha / ((N0**2)-N0)/2 # Compute the new significance level.
    
    corrmat = np.zeros((N0, N0)) # Initialize various matrices
    pmat_corr = np.zeros((N0, N0))
    
    for i in range(N0): # Every unique combination of channels
        for j in range(i):
            rho, p = pearsonr(gsr[i], gsr[j]) # Pearsons rho
            corrmat[i][j] = rho # Make sure to add both upper and lower triangles
            corrmat[j][i] = rho
            
            pmat_corr[i][j] = p
            pmat_corr[j][i] = p
    
    # Turn Pearson correlation coeffs into MI in nats. 
    mimat = -np.log(1 - (corrmat**2))/2 
    
    # Significance testing.
    corrmat_correct = corrmat*(pmat_corr < bonferonni)
    mimat_correct = mimat*(pmat_corr < bonferonni)
    
    return corrmat, mimat, corrmat_correct, mimat_correct, pmat_corr


def whiten_signal(gsr, hist=1):
    """
    Whitens the signals by removing the autocorrelation. 
    Inspiried by: 
        Daube, C., Gross, J., & Ince, R. A. A. (2022). 
        A whitening approach for Transfer Entropy permits the application to narrow-band signals. 
        ArXiv:2201.02461 [q-Bio]. http://arxiv.org/abs/2201.02461

    Parameters
    ----------
    gsr : np.ndarray
        The GSR'd data.
    hist : int
        The number of bins of history to account for when computing entropy rate.

    Returns
    -------
    entrate : np.ndarray
        The whitened signals
    """
    
    
    gauss_class = JPackage("infodynamics.measures.continuous.gaussian").EntropyCalculatorGaussian
    gauss_estimator = gauss_class()

    ais_class = JPackage("infodynamics.measures.continuous.gaussian").ActiveInfoStorageCalculatorGaussian
    ais_estimator = ais_class()
    
    entrate = np.zeros((N0, N1-hist)) # Initialize the entropy rate array.
    
    for i in range(N0): # For each channel
        
        # Compute entropy of the channel
        gauss_estimator.initialise()
        gauss_estimator.setObservations(gsr[i][:].tolist())
        gauss_result = np.array(
                gauss_estimator.computeLocalOfPreviousObservations()
            )
        
        # Compute the active information storage of the channel
        ais_estimator.initialise()
        ais_estimator.setObservations(gsr[i][:].tolist())
        ais_result = np.array(
            ais_estimator.computeLocalOfPreviousObservations()
            )
    
        entrate[i,:] = gauss_result[1:] - ais_result[1:] # Slicing off the first element since it's 0 for AIS (no history).
    
    return entrate


def time_lagged_mutual_information(entrate, alpha):
    """
    Computes the time-lagged mutual information matrix 
    from the whitened signals

    Parameters
    ----------
    entrate : np.ndarray
        The whitened data.
    alpha : float
        The signifiance threshold.

    Returns
    -------
    effmat_correct : np.ndarray
        The directed effective connectivity mat.
    lagmat_correct : np.ndarray
        The lag that maximizes the temporal mutual information.

    """
    
    N0 = entrate.shape[0]
    
    mi_tensor = np.zeros((N0, N0, 10))
    p_tensor = np.zeros((N0, N0, 10))
    
    for i in range(mi_tensor.shape[-1]): # For every lag
        for j in range(N0): # Filling the upper triangle
            for k in range(N0): # Filling the upper triangle
                rho, p = pearsonr(entrate[j][:-(i+1)], # Gnarly indexing.
                                  entrate[k][i+1:])
                
                mi_tensor[j,k,i] = -np.log(1-(rho**2))/2 # Converting Pearson correlation to MI
                p_tensor[j,k,i] = p
    
    bonferonni_tensor = alpha/((N0**2) * 10)
    mi_tensor_correct = mi_tensor * (p_tensor < bonferonni_tensor)

    effmat_correct = np.max(mi_tensor_correct, axis=-1)
    lagmat_correct = np.argmax(mi_tensor_correct, axis=-1) + 1
    
    return mi_tensor, p_tensor, effmat_correct, lagmat_correct


if __name__ == '__main__':

    #%%
    start_jvm("/home/thosvarley/.bin/jidt/infodynamics.jar")

    alpha = 0.001 # Uncorrected p-value

    in_dir = '/home/thosvarley/Documents/indiana_university/research/xenobots/data/'
    listdir = sorted(os.listdir(in_dir))

    listdir = [x for x in listdir if int(x.split("_")[1]) in {x for x in range(7,12)}]

    out_dir = "/home/thosvarley/Documents/indiana_university/research/xenobots/results/"
    fig_dir = "/home/thosvarley/Documents/indiana_university/research/xenobots/results/figures/"

    #%%
    for bot in listdir: 
        if bot == bot: #"bot_10":
            print(bot)
            #%%
            # This read/write structure is specific to how I saved the files,
            # You can change it to match your own directory structure (w/e it is). 
            
            listdir_bot = os.listdir(in_dir + bot)
            
            for scan in listdir_bot:
                
                df = pd.read_csv(in_dir + "{0}/{1}/series.csv".format(bot, scan), header=0)
                
                ids = df["id"].unique()
                time = df["timestep"].max()
                
                c = df["value"].values
                print(scan)
                series = c.reshape((ids.shape[0], int(c.shape[0]/ids.shape[0])))
                print(series.shape)
                
                series, rejected_frames = artifact_filter(series, threshold=5)
                print(series.shape)
                np.savetxt(out_dir + "rejected_frames_{0}_{1}.csv".format(bot, scan), rejected_frames, delimiter=",")
                
                N0 = series.shape[0]
                N1 = series.shape[1] 
                
                #%%
                # Some time series are all one value - this messes things up,
                # so we filter out those cells. 
                series = series[~np.isclose(series.var(axis=-1), 0),:] 
                
                np.savetxt(out_dir + "series_{0}_{1}_filter.csv".format(bot, scan), series, delimiter=",")
                
                # GSR
                # This could also be done w/ local conditional entropies, probably.
                # Worth exploring later? 
                
                gsr = global_signal_regression(series)
                np.savetxt(out_dir + "series_gsr_{0}_{1}_filter.csv".format(bot, scan), gsr, delimiter=",")
            
                # Basic FC w/ Bonferonni-corrected significance testing
        
                corrmat, mimat, corrmat_correct, mimat_correct, pmat_corr = functional_connectivity(gsr, alpha=alpha)
                
                np.savetxt(out_dir + "fc_rho_gsr_{0}_{1}_filter.csv".format(bot, scan), corrmat, delimiter=",")
                np.savetxt(out_dir + "fc_mi_gsr_{0}_{1}_filter.csv".format(bot, scan), mimat, delimiter=",")
                np.savetxt(out_dir + "fc_pval_gsr_{0}_{1}_filter.csv".format(bot, scan), pmat_corr, delimiter=",")
                
                np.savetxt(out_dir + "fc_rho_gsr_sigthresh_{0}_{1}_filter.csv".format(bot, scan), corrmat_correct, delimiter=",")
                np.savetxt(out_dir + "fc_mi_gsr_sigthresh_{0}_{1}_filter.csv".format(bot, scan), mimat_correct, delimiter=",")
                #%%
                # Compute lag-1 local entropy rates from GSR signal.
                # Whitens series by removing autocorrelation. 
                # Entropy rate is computed as the difference between h(future) - i(past ; future)
                
                hist = 1
                
                entrate = whiten_signal(gsr, hist=hist)
                
                np.savetxt(out_dir + "series_whiten_{0}_{1}_filter.csv".format(bot, scan), entrate, delimiter=",")
                #%%
                # Basic FC for whitened, entropy-rate signals. 
                
                corrmat_entrate, mimat_entrate, corrmat_entrate_correct, mimat_entrate_correct, pmat_entrate = functional_connectivity(entrate, alpha=alpha)
                        
                np.savetxt(out_dir + "fc_mi_whiten_{0}_{1}_filter.csv".format(bot, scan), mimat_entrate, delimiter=",")
                np.savetxt(out_dir + "fc_pval_whiten_{0}_{1}_filter.csv".format(bot, scan), pmat_entrate, delimiter=",")
                np.savetxt(out_dir + "fc_mi_whiten_sigthresh_{0}_{1}_filter.csv".format(bot, scan), mimat_entrate_correct, delimiter=",")
                
                # Computing the time-lagged mutual information for the whitened time series.
                
                mi_tensor, p_tensor, effmat_correct, lagmat_correct = time_lagged_mutual_information(entrate, alpha=alpha)
                
                np.savez_compressed(out_dir + "tensor_mi_whiten_{0}_{1}_filter.npz".format(bot, scan), mi_tensor)
                np.savez_compressed(out_dir + "tensor_pvals_whiten_{0}_{1}_filter.npz".format(bot, scan), p_tensor)
                
                np.savetxt(out_dir + "ec_mi_whiten_sigthresh_{0}_{1}_filter.csv".format(bot, scan), effmat_correct, delimiter=",")
                np.savetxt(out_dir + "ec_lags_whiten_sigthresh_{0}_{1}_filter.csv".format(bot, scan), lagmat_correct, delimiter=",")
                
                #%% Matplotlib figure.
                
                fig = plt.figure(figsize=(10,8), dpi=400)
                gs = plt.GridSpec(4, 4, figure=fig)
                
                ax1 = plt.subplot(gs[0,:])
                ax1.set_title("Global Signal-Regressed Calcium Signal")
                ax1.set_ylabel("Cells")
                ax1.set_xlabel("Time")
                
                ax2 = plt.subplot(gs[1,:])
                ax2.set_title("Whitened GSR Signal (Ent Rate Lag {0})".format(hist))
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Cells")
                
                ax3 = plt.subplot(gs[2:,:2])
                ax3.set_title("Gaussian MI (GSR)")
                
                ax4 = plt.subplot(gs[2:,2:])
                ax4.set_title("Gaussian MI (GSR + Whitened)")
                
                imshow1 = ax1.imshow(gsr[:,hist:], 
                                    aspect="auto", 
                                    cmap="inferno",
                                    vmin=gsr.min(),
                                    vmax=gsr.max())
                imshow2 = ax2.imshow(entrate, 
                                    aspect="auto", 
                                    cmap="inferno",
                                    vmin=0,
                                    vmax=gsr.max())
                
                mimat_correct[mimat_correct == 0] = np.nan
                mimat_entrate_correct[mimat_entrate_correct == 0] = np.nan
                
                imshow3 = ax3.imshow(mimat_correct)
                imshow4 = ax4.imshow(mimat_entrate_correct)
                
                plt.colorbar(imshow3, ax=ax3, label="Nat")
                plt.colorbar(imshow4, ax=ax4, label="Nat")
                
                plt.tight_layout()
                plt.savefig(fig_dir + "whitening_gsr_{0}_{1}_filter.png".format(bot, scan), 
                            bbox_inches="tight")
                
                del df, mimat_correct, mimat_entrate_correct