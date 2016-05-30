%% DNN Test
%create a Deep neural network based on the passed parameters from Python
%theano
clear; close; clc
load ToTheano1min_5.mat
load dnnModel_1min_5_best.mat
numLayers = 5;

%% Define network structure
layers = struct;
layers(1).W = tanhLayer0_W;
layers(1).b = tanhLayer0_b;
layers(2).W = tanhLayer1_W;
layers(2).b = tanhLayer1_b;
layers(3).W = tanhLayer2_W;
layers(3).b = tanhLayer2_b;
layers(4).W = tanhLayer3_W;
layers(4).b = tanhLayer3_b;
layers(5).W = outputLayer_W;
layers(5).b = outputLayer_b;

%% Feed-Forward on the train set
train_feat = cell(numLayers+1,1);
train_feat{1} = train_x';
train_label = train_y+1;
for i=1:numLayers
    if (i<numLayers)
        [train_feat{i+1}] = TanhFullyConnectedAct(train_feat{i}, layers(i).W, layers(i).b);
    else
        [train_feat{i+1}] = SoftmaxFullyConnectedAct(train_feat{i}, layers(i).W, layers(i).b);        
    end
end
train_pred = train_feat{numLayers+1}';
figure('color','w')
imagesc(train_pred);

disp('Classification accuracy')
[~,train_group] = max(train_pred,[],1);
disp(sum(train_group==train_label)/length(train_label))

%% Feed-Forward on the valid set
valid_feat = cell(numLayers+1,1);
valid_feat{1} = valid_x';
valid_label = valid_y+1;
for i=1:numLayers
    if (i<numLayers)
        [valid_feat{i+1}] = TanhFullyConnectedAct(valid_feat{i}, layers(i).W, layers(i).b);
    else
        [valid_feat{i+1}] = SoftmaxFullyConnectedAct(valid_feat{i}, layers(i).W, layers(i).b);        
    end
end
valid_pred = valid_feat{numLayers+1}';
figure('color','w')
imagesc(valid_pred);

disp('Classification accuracy')
[~,valid_group] = max(valid_pred,[],1);
disp(sum(valid_group==valid_label)/length(valid_label))

%% Feed-Forward on the test set
test_feat = cell(numLayers+1,1);
test_feat{1} = test_x';
test_label = test_y+1;
for i=1:numLayers
    if (i<numLayers)
        [test_feat{i+1}] = TanhFullyConnectedAct(test_feat{i}, layers(i).W, layers(i).b);
    else
        [test_feat{i+1}] = SoftmaxFullyConnectedAct(test_feat{i}, layers(i).W, layers(i).b);        
    end
end
test_pred = test_feat{numLayers+1}';
figure('color','w')
imagesc(test_pred);

disp('Classification accuracy')
[~,test_group] = max(test_pred,[],1);
disp(sum(test_group==test_label)/length(test_label))



%% Test up/down accuracy
load 1min_5_percentile.mat
perc = [percentile(1:2),0,percentile(3:4)];
test_predValue = perc*test_pred;
load 1min_5_label.mat
test_realValue = label(round(num_sample*0.9)+1:end);

sum(test_predValue.*test_realValue>=0)/length(test_realValue)

%% Test pnl
mean(test_realValue(test_predValue>=0))*100
mean(test_realValue(test_predValue<0))*-100

cumul_pnl = zeros(length(test_predValue),1); cumul_pnl(1)=500;
for cnt =2:length(test_realValue)
    if test_predValue(cnt)>=0
        cumul_pnl(cnt) = 1/5*cumul_pnl(cnt-1)*(1+test_realValue(cnt))+4/5*cumul_pnl(cnt-1);
    elseif test_predValue(cnt)<0
        cumul_pnl(cnt) = 1/5*cumul_pnl(cnt-1)*(1-test_realValue(cnt))+4/5*cumul_pnl(cnt-1);
    end
end
figure('color','w')
plot(cumul_pnl);xlabel('sample index');ylabel('cumul-pnl');title('Test set')
set(gca,'fontsize',20)