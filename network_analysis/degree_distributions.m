clearvars
close all
clc

% Requires the Brain Connectivity Toolbox
% https://sites.google.com/site/bctnet/getting-started

addpath('./BCT/2019_03_03_BCT/');


bots = {'bot_01', 'bot_02', 'bot_03', 'bot_04', 'bot_05', 'bot_06'};


for i = 1:length(bots)

    bot = string(bots(i));
    
    % read in functional connectivity matrix
    FC1 = readmatrix(strcat('../network_inference_data/fc_matrices/', bot, '_before_mimat_w.csv'));
    FC2 = readmatrix(strcat('../network_inference_data/fc_matrices/', bot, '_after_mimat_w.csv'));

    % before
    deg1 = sort(degrees_und(FC1), 'descend');
    writematrix(deg1, strcat('../network_analysis_data/degree_distributions/',bot,'_before_degree_distribution.csv'))
    
    % after
    deg2 = sort(degrees_und(FC2), 'descend');
    writematrix(deg2, strcat('../network_analysis_data/degree_distributions/',bot,'_after_degree_distribution.csv'))
    
end