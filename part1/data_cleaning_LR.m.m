%% Cubist Project
% This algorithm is written as part of the solving process of the cubist
% project. Written by Minda Yang: minda.yang@columbia.edu
% 
% This part includes data visualization, data cleaning and basic linear
% regression algorithm
clear all; close all; clc;


%%%%%%% Part 1: Load data and visualize the data %%%%%%%
csv_file = csvread('data.csv',1,1);
% figure('color','w')
% plot(csv_file(:,1));hold on;plot(csv_file(:,2));
% figure('color','w');plot(csv_file(:,1));
% figure('color','w');plot(csv_file(:,2),'r');
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
signal = csv_file(:,1);    %the signal
spy_close = csv_file(:,2); %the close price of spy
% take notes of the index that is cleaned
signal_clean_note = [];
spy_clean_note = [];
%loop across the values and clean both data
for cnt = 22:length(signal)
    %calculate the difference
    signal_diff = signal(cnt-20:cnt-1)-signal(cnt-21:cnt-2);
    spy_diff = spy_close(cnt-20:cnt-1)-spy_close(cnt-21:cnt-2);
    %mean and std
    signal_diff_mean = mean(signal_diff);
    signal_diff_std = std(signal_diff);
    spy_diff_mean = mean(spy_diff);
    spy_diff_std = std(spy_diff);
    %data cleaning
    if signal(cnt)-signal(cnt-1) > signal_diff_mean+10*signal_diff_std ...
            || signal(cnt)-signal(cnt-1) < signal_diff_mean-10*signal_diff_std
        signal(cnt) = signal(cnt-1);
        signal_clean_note = [signal_clean_note,cnt];
    end
    if spy_close(cnt)-spy_close(cnt-1) > spy_diff_mean+10*spy_diff_std ...
            || spy_close(cnt)-spy_close(cnt-1) < spy_diff_mean-10*spy_diff_std
        spy_close(cnt) = spy_close(cnt-1);
        spy_clean_note = [spy_clean_note,cnt];
    end
end
figure('color','w');plot(signal); title('signal after data cleaning')
figure('color','w');plot(spy_close,'r'); title('spy close after data cleaning')
% z-score the signal and spy_close
tmp1 = zscore(signal);
tmp2 = zscore(spy_close);
figure;plot(tmp1);hold on; plot(tmp2,'r');legend({'spy-close','signal'})
%%%%%%%% End of Part 2 %%%%%%%



%%%%%%% Part 3: Linear regression algorithm %%%%%%%
% Basic logic:
%   We receive both data at the same time, so it only makes sense to use
%   today's signal (and signals before today) to predict future's return.
%   To assess whether the signal has predictive power, here I compare the
%   predicitve power of using signal vs. predictive power using spy index. 
%   
%   For example, to assess whether signal(1:20) can predict spy(21) better 
%   than spy(1:20). Further, whether combining signal(1:20) and spy(1:20)
%   performs better than spy(1:20) alone.
%
% Notice:
%   To predict return, we need to calculate the difference between today's
%   close vs. tomorrows close.
%   To fairly compare two approaches (using signal vs. using spy) to
%   predict future spy. We need to train a model in the trainig set and
%   compare the prediction in the test set.
%   To fully take advantage of the data, we can do this in a
%   cross-validation manner. But in this demo I'll specify the training set
%   and testing set.
% Future:
%   Possibly linear regression with some regularization.

%prepare the data
train_x = zeros(21,400);   %using past 20days of signal return and one constant, train with first 400 samples
train_x2 = zeros(21,400);  %using past 20days of spy return and one constant, train with first 400 samples
train_x3 = zeros(41,400);  %using both signal and spy, to test whether add value
train_y = zeros(1,400);    %the real value of spy return for the training set

test_x = zeros(21,246);    %using past 20days of signal return and one constant, test with left samples
test_x2 = zeros(21,246);   %using past 20days of spy return and one constant, test with left samples
test_x3 = zeros(41,246);   %using both signal and spy, to test whether add value
test_y = zeros(1,246);     %the real value of spy return for the test set
%train_set
for cnt = 22:421
    train_x(:,cnt-21) = [(signal(cnt-20:cnt-1)-signal(cnt-21:cnt-2))./signal(cnt-21:cnt-2);1];
    train_x2(:,cnt-21) = [(spy_close(cnt-20:cnt-1)-spy_close(cnt-21:cnt-2))./spy_close(cnt-21:cnt-2);1];
    train_x3(:,cnt-21) = [(signal(cnt-20:cnt-1)-signal(cnt-21:cnt-2))./signal(cnt-21:cnt-2);...
        (spy_close(cnt-20:cnt-1)-spy_close(cnt-21:cnt-2))./spy_close(cnt-21:cnt-2);1];
    train_y(cnt-21) = (spy_close(cnt)-spy_close(cnt-1))/spy_close(cnt-1);
end
%test set
for cnt=422:667
    test_x(:,cnt-421) = [(signal(cnt-20:cnt-1)-signal(cnt-21:cnt-2))./signal(cnt-21:cnt-2);1];
    test_x2(:,cnt-421) = [(spy_close(cnt-20:cnt-1)-spy_close(cnt-21:cnt-2))./spy_close(cnt-21:cnt-2);1];
    test_x3(:,cnt-421) = [(signal(cnt-20:cnt-1)-signal(cnt-21:cnt-2))./signal(cnt-21:cnt-2);...
        (spy_close(cnt-20:cnt-1)-spy_close(cnt-21:cnt-2))./spy_close(cnt-21:cnt-2);1];
    test_y(:,cnt-421) = (spy_close(cnt)-spy_close(cnt-1))/spy_close(cnt-1);
end
%linear regression
w = regress(train_y',train_x');
w2 = regress(train_y',train_x2');
w3 = regress(train_y',train_x3');
%calculate the prediction using the weights found
pred = test_x'*w;
pred2 = test_x2'*w2;
pred3 = test_x3'*w3;
figure('color','w')
plot(test_y,'k','linewidth',2);hold on;plot(pred,'b--','linewidth',2),plot(pred2,'r.-','linewidth',2),plot(pred3,'g','linewidth',2)
legend({'real value','prediction-signal','prediction-spy','prediction-combine'})
set(gca,'fontsize',20)

disp('Mean squared error (using signal): '); disp((sum((pred-test_y').^2)/length(test_y)))
disp('Mean squared error (using spy): '); disp((sum((pred2-test_y').^2)/length(test_y)))
disp('Mean squared error (combining signal and spy): '); disp((sum((pred3-test_y').^2)/length(test_y)))

%%%%%%% Part 4: save the training and testing data for DNN %%%%%%%
Train_X1 = train_x2(1:20,:);
Train_X2 = train_x3(1:40,:);
Train_Y = train_y*40;
Valid_X1 = test_x2(1:20,1:123); % split to half, valid and test
Valid_X2 = test_x3(1:40,1:123);
Valid_Y = test_y(1:123)*40;
Test_X1 = test_x2(1:20,124:end); 
Test_X2 = test_x3(1:40,124:end);
Test_Y = test_y(124:end)*40;

save ToTheano.mat Train_X1 Train_X2 Train_Y Valid_X1 Valid_X2 Valid_Y Test_X1 Test_X2 Test_Y -v7.3
