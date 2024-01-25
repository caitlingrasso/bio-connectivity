clearvars
close all
clc

% Requires the Brain Connectivity Toolbox
% https://sites.google.com/site/bctnet/getting-started

addpath('./BCT/2019_03_03_BCT/');


bots = {'bot_01', 'bot_02', 'bot_03', 'bot_04', 'bot_05', 'bot_06'};
phases = {'after'};

path = '../network_inference_data/fc_matrices_synthetic/';
label = 'mimat_w_nospikes';

outpath = '../network_analysis_data/degree_distributions/';

for i = 1:length(bots)

    for j = 1:length(phases)

        phase = string(phases(j));

        bot = string(bots(i));
        
        % read in functional connectivity matrix
        FC = readmatrix(strcat(path, bot, '_', phase,'_',label,'.csv'));

        % after
        deg = sort(degrees_und(FC), 'descend');
        writematrix(deg, strcat(outpath ,bot,'_', phase,'_',label,'_degree_distribution.csv'))

    end
    
end