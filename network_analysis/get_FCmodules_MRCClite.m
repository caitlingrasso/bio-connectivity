function [ciu,Aall,anull,A,ciall] = get_FCmodules_MRCClite(K,sam1,sam2,maxC,r1,r2)

% performs a 'lite' version of MRCC, obtaining a CC (Aall) matrix and a set of
% consensus modules

N = size(K,1);

% FIRST PASS - find gamma range where modules range from 2 to maxC;
% initial gamma range
gam_range = linspace(r1,r2,sam1);
G = length(gam_range);

ciall = zeros(N,G);
for g=1:length(gam_range)
    [ci, ~] = community_louvain2(K,gam_range(g),[],'potts');
    ciall(:,g) = ci;
end

% identify the limits of the range
ff = find((max(ciall)>1)&(max(ciall)<maxC));
g1 = gam_range(ff(1));
g2 = gam_range(ff(end));
disp([num2str(g1),' ',num2str(g2)])

% SECOND PASS - use the gamma range determined in first pass
% collect all partitions within that range
gam_range = linspace(g1,g2,sam2);
G = length(gam_range);

ciall = zeros(N,G);
for g=1:G
    [ci,~] = community_louvain2(K,gam_range(g),[],'potts');
    ciall(:,g) = ci;
end

% exclude spurious partitions that are outside of the range 2 to N
numm = max(ciall);
use = find((numm>1)&(numm<maxC));
ciall = ciall(:,use);

% co-classification matrix
Aall = agreement(ciall)./length(use);

% analytic null
anull = 0;
for cnum = 1:size(ciall,2)
    anull = anull+sum((histcounts(ciall(:,cnum))./N).*((histcounts(ciall(:,cnum))-1)./(N-1)));
end
A = Aall - anull/length(use);

% consensus clustering
tau = 0;
ciu = consensus_und(A,tau,10);
