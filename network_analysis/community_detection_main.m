clearvars
close all
clc

% Requires the Brain Connectivity Toolbox
% https://sites.google.com/site/bctnet/getting-started

addpath('./BCT/2019_03_03_BCT/');

% read in functional connectivity matrix
bots = {'bot_02', 'bot_03', 'bot_04', 'bot_05', 'bot_06'};
phases={'after'};

path = '../network_inference_data/fc_matrices_synthetic/';
label = 'mimat_w_nospikes';

outpath = '../network_analysis_data/community_detection/';

for i = 1:length(bots)
    for j=1:length(phases)

        bot = string(bots(i));
        phase = string(phases(j));
    
        % read in functional connectivity matrix
        FC = readmatrix(strcat(path, bot, '_',phase,'_',label,'.csv'));
        
        N = length(FC); % # of nodes in the FC matrix
        
        % compute partitions
        sam1 = 100;
        sam2 = 1000;
        maxC = N;
        r1 = -1; r2 = 1;
        
        [ciu,Aall,anull,A,~] = get_FCmodules_MRCClite(FC,sam1,sam2,maxC,r1,r2);
        
        writematrix(ciu, strcat(outpath, bot, '_',phase,'_',label,'_consensus_partition.csv'))
        writematrix(A, strcat(outpath, bot, '_',phase,'_',label,'_consensus_matrix.csv'))
    end
end


