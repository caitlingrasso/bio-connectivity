clear
clc

% load time series
bot = 'bot_01';
before_label='\bfO1-pre';
after_label='\bfO1-post';

TSw1 = readmatrix(strcat('../network_inference_data/series_gsr_whitened/', bot, '_before_entrate.csv'));
TSw2 = readmatrix(strcat('../network_inference_data/series_gsr_whitened/', bot, '_after_entrate.csv'));

lts1 = size(TSw1,2);
lts2 = size(TSw2,2);

load mycmap

% module colors
cmapm = [1 0 0; 0 1 0; 0 0 1; 1 1 0; 1 0 1; 0 1 1; ...
    0.5 0 1; 0 0.5 1; 0.5 0 0; 0.2 0.2 0.2; 0 0 0.5; 0.2 0.75 0.6; ...
    0.2 0.4 0.6; 0.6 0.4 0.2; 0.1 0.6 0.3; 0.9 0.3 0.7];
cmapm = rand(50,3);

% ets (whiten)
zTS1 = zscore(TSw1');
ets1 = fcn_edgets(zTS1);
rss1 = sum(ets1.^2,2).^0.5;

zTS2 = zscore(TSw2');
ets2 = fcn_edgets(zTS2);
rss2 = sum(ets2.^2,2).^0.5;

figure('position',[0 0 600 500]); % was [100 100 800 800]
subplot(2,2,1)

imagesc(ets1',[-1 1])
% xlabel('Time', fontsize=20); 
ylabel('Edges', fontsize=20); title(before_label, fontsize=25)

ax=gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;

subplot(2,2,2)
imagesc(ets2',[-1 1])
% xlabel('Time', fontsize=20); ylabel('Edges', fontsize=20); 
title(after_label, fontsize=25)
ax=gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;

subplot(2,2,3)
hold on
ar = area(rss1); set(ar,'FaceColor',[0.8 0.8 1]);
plot(rss1,'b'); axis([1 lts1 0 max([max(rss1) max(rss2)])]);
xlabel('Time', fontsize=20); ylabel('RSS Amplitude', fontsize=20)
box on
ax=gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;

subplot(2,2,4)
hold on
ar = area(rss2); set(ar,'FaceColor',[0.8 0.8 1]);
plot(rss2,'b'); axis([1 lts2 0 max([max(rss1) max(rss2)])]);
xlabel('Time', fontsize=20); 
% ylabel('RSS Amplitude', fontsize=20)
box on
ax=gca;
ax.XAxis.FontSize = 20;
ax.YAxis.FontSize = 20;


colormap(flipud(mycmap))

saveas(gcf, strcat('../results/edge_timeseries/', bot, '_ets.png'))
