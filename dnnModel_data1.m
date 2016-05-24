%% DNN Test
%create a Deep neural network based on the passed parameters from Python
%theano
clear; close; clc
load ToTheano2.mat
load dnnModel_data_Train1.mat
numLayers = 2;

%% Define network structure
layers = struct;
layers(1).W = double(tanhLayer0_W);
layers(1).b = double(tanhLayer0_b);
%layers(2).W = tanhLayer1_W;
%layers(2).b = tanhLayer1_b;
layers(2).W = double(outputLayer_W);
layers(2).b = double(outputLayer_b);

%% Feed-Forward on the train set
train_feat = cell(numLayers+1,1);
train_feat{1} = Train_X1';
train_label = Train_Y;
for i=1:numLayers
    if (i<numLayers)
        [train_feat{i+1}] = TanhFullyConnectedAct(train_feat{i}, layers(i).W, layers(i).b);
    else
        [train_feat{i+1}] = TanhFullyConnectedAct(train_feat{i}, layers(i).W, layers(i).b);        
    end
end
train_label = train_label./40;
train_pred = train_feat{numLayers+1}'./(40*std(train_feat{numLayers+1})/std(Train_Y));
figure('color','w')
plot(train_pred);hold on;
plot(train_label,'r'); 

disp((sum((train_pred-train_label).^2)/length(train_label)))
disp('Classification accuracy')
disp(sum(train_pred.*train_label>0)/length(train_label))

%% Feed-Forward on the valid set
valid_feat = cell(numLayers+1,1);
valid_feat{1} = Valid_X1';
valid_label = Valid_Y;
for i=1:numLayers
    if (i<numLayers)
        [valid_feat{i+1}] = TanhFullyConnectedAct(valid_feat{i}, layers(i).W, layers(i).b);
    else
        [valid_feat{i+1}] = TanhFullyConnectedAct(valid_feat{i}, layers(i).W, layers(i).b);        
    end
end
valid_label = valid_label./40;
valid_pred = valid_feat{numLayers+1}'./(40*std(valid_feat{numLayers+1})/std(Valid_Y));
figure('color','w')
plot(valid_pred);hold on;
plot(valid_label,'r'); 

disp((sum((valid_pred-valid_label).^2)/length(valid_label)))
disp('Classification accuracy')
disp(sum(valid_pred.*valid_label>0)/length(valid_label))

%% Feed-Forward on the test set
test_feat = cell(numLayers+1,1);
test_feat{1} = Test_X1';
test_label = Test_Y;
for i=1:numLayers
    if (i<numLayers)
        [test_feat{i+1}] = TanhFullyConnectedAct(test_feat{i}, layers(i).W, layers(i).b);
    else
        [test_feat{i+1}] = TanhFullyConnectedAct(test_feat{i}, layers(i).W, layers(i).b);        
    end
end
test_label = test_label./40;
test_pred = test_feat{numLayers+1}'./(40*std(valid_feat{numLayers+1})/std(Valid_Y));
figure('color','w')
plot(test_pred);hold on;
plot(test_label,'r'); legend({'Prediction-Signal','Real value'});set(gca,'fontsize',20)

disp('Mean squared error (using signal): ');
disp((sum((test_pred-test_label).^2)/length(test_label)))

disp('Correlation using signal')
disp(corrcoef(test_pred,test_label))

disp('Classification accuracy')
disp(sum(test_pred.*test_label>0)/length(test_label))



