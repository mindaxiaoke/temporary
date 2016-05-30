%% Cubist Project
% This algorithm is written as part of the solving process of the cubist
% project. Written by Minda Yang: minda.yang@columbia.edu
% 
% This part includes data visualization, data cleaning and basic linear
% regression algorithm
clear all; close all; clc;


%%%%%%% Part 1: Load data and visualize the data %%%%%%%
csv_file = readtable('data.1_min.csv', 'ReadVariableNames', false, 'ReadRowNames', false);
%%%%%%% End of Part 1 %%%%%%%



%%%%%%% Part 2: Data cleaning %%%%%%%
% Basic logic:
%   Look at the previous 20 days, calculate the average and standard deviation of
%   difference between 2 consecutive days.
%   If the new sample is out of the 10*stardard deviation range, then
%   consider it to be noisy and replace with the previous day's value.
% Notice:
%   This method assumes the distribution of the difference to be Gaussian,
%   which is not necessarily true, the usage of 10*stardard deviation as 
%   a large margin here is to try to deal with the 'fat tail' effect.
% Future:
%   Add a method to examine the distribution of the difference.
%   Possibly LPC (linear predictive coding)

% copy the data
csv_cell = table2cell(csv_file);
date = csv_cell(:,1);
signal = cell2mat(csv_cell(:,2));    %the signal
% take notes of the index that is cleaned
signal_clean_note = [];
%loop across the values and clean signal
diff_signal = (signal(2:end)-signal(1:end-1));
bad_index = find(diff_signal > mean(diff_signal)+20*std(diff_signal) ...
            | diff_signal < mean(diff_signal)-20*std(diff_signal));
while ~isempty(bad_index) && length(bad_index)<20
    signal(bad_index(1)+1)=signal(bad_index(1));
    signal_clean_note = [signal_clean_note,bad_index(1)];
    diff_signal = (signal(2:end)-signal(1:end-1));
    bad_index = find(diff_signal > mean(diff_signal)+20*std(diff_signal) ...
        | diff_signal < mean(diff_signal)-20*std(diff_signal));
%     if length(bad_index)>=20
%         error('Too many noisy samples, recommend manual cleaning');
%     end
end
% figure('color','w');plot(signal); title('signal after data cleaning')


%%%%%%% Part 3: Prepare and reshape data %%%%%%%
% Basic logic:
%   The over-night price change should be considered separately
%   Reshape the data/signal into date by minute

% ret_signal = (signal(2:end)-signal(1:end-1))./signal(1:end-1);
% figure;plot(ret_signal);legend({'Ret of security'})
date_mat = {};  % the cell for all dates
signal_mat = {}; % the cell for all signal
cur_date = [];   % the cell for current time (one day)
cur_signal = []; % the cell for current signal (one day)
today = date{1}(1:10);  % extract the current date
cur_time = str2double(date{1}(12:13))*60+str2double(date{1}(15:16));
time = str2double(date{1}(12:13))*60+str2double(date{1}(15:16))-1; % initial time set to (-1)
cnt = 1;
while cnt <= length(signal)
    while abs(cur_time - time) <= 30 % only allow 30 minutes maximal interval
        while cur_time > time + 1
            time = time+1; % missing one minute, so add time by 1
            cur_signal = [cur_signal,cur_price];
        end
        cur_price = signal(cnt);
        cur_signal = [cur_signal,cur_price];
        cnt = cnt+1;
        if cnt == 1586545
            A = 1;
        end
        if cnt>length(signal)
            break
        end
        cur_time = str2double(date{cnt}(12:13))*60+str2double(date{cnt}(15:16));
        time = time+1;
        if time == 1439 %(23*60+59)
            time = -1;
        end
    end
    signal_mat = [signal_mat;{cur_signal}];
    if cnt<=length(signal)
        today = date{cnt}(1:10);  % extract the current date
        time = str2double(date{cnt}(12:13))*60+str2double(date{cnt}(15:16))-1;
        cur_signal = [];
        cur_date = [];
    end
    cnt
end
%%%%%%% End of Part 3 %%%%%%%

%%%%%%% Part 4: Exclude bad samples and prepare data for DNN %%%%%%%
% load 1min_parseData.mat
sample_length = zeros(size(signal_mat));
for cnt = 1:size(signal_mat,1)
    sample_length(cnt)=length(signal_mat{cnt});
end
%figure('color','w');histogram(sample_length,30);set(gca,'fontsize',20);
signal_subset = signal_mat(sample_length>90);
% use 30 mins data and the high, low, avg of previous time as feature
% logic:
%    Use the previous 30 mins of price, together with the high, low and avg
%    price of the current continuous data as feature
%    Assume execution in one minute (31 min) and the pnl for the next
%    nine minute(40 min)
num_sample = sum(sample_length(sample_length>90))-34*size(signal_subset,1); % total number of samples
feat = zeros(33,num_sample); % initialize features matrix
label = zeros(1,num_sample); % initialize return vector
sample_index = 1;
for cnt1 = 1:size(signal_subset,1)
    for cnt2 = 1:length(signal_subset{cnt1})-34
        tmp = signal_subset{cnt1}; % the current continuous price block
        serie = [0,(tmp(2:end)-tmp(1:end-1))./tmp(1:end-1)]; % the return serie
        cur_feat = [serie(cnt2:cnt2+29),max(tmp(1:cnt2+29))/mean(tmp(cnt2:cnt2+29)),min(tmp(1:cnt2+29))/mean(tmp(cnt2:cnt2+29)),mean(tmp(1:cnt2+29))/mean(tmp(cnt2:cnt2+29))]';
        cur_label = (tmp(cnt2+34)-tmp(cnt2+30))/tmp(cnt2+30);
        feat(:,sample_index) = cur_feat;
        label(sample_index) = cur_label;
        sample_index=sample_index+1;
    end
    cnt1
end
%group = ordinal(label,{'1','2','3','4','5'},[],prctile(label,[0,20,40,60,80,100]));
%group = int8(group);
% convert return distribution to groups for classification
percentile = prctile(label,[20,40,60,80]);
group = zeros(size(label));
group(label<percentile(1)) = 0; % group LL (worse return)
group(label>=percentile(1) & label<percentile(2)) = 1; % group L (bad return)
group(label>=percentile(2) & label<=percentile(3)) = 2; % group F (flat return)
group(label>percentile(3) & label<=percentile(4)) = 3; % group H (good return)
group(label>=percentile(4)) = 4; % group HH (better return)

% normalize each feature dimension
[feat_zscore,mu,sigma] = zscore(feat,1,2);

% using a 10-fold to separate train, valid and test
train_x = feat_zscore(:,1:round(num_sample*0.8));
train_y = group(1:round(num_sample*0.8));
valid_x = feat_zscore(:,round(num_sample*0.8)+1:round(num_sample*0.9));
valid_y = group(round(num_sample*0.8)+1:round(num_sample*0.9));
test_x = feat_zscore(:,round(num_sample*0.9)+1:end);
test_y = group(round(num_sample*0.9)+1:end);

save ToTheano1min_5.mat train_x train_y valid_x valid_y test_x test_y -v7.3



        
        
        