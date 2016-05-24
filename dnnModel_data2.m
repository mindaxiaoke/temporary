%% DNN Test
%create a Deep neural network based on the passed parameters from Python
%theano
clear; close; clc
load ToTheano.mat
load dnnModel_data2.mat
numLayers = 2;

%% Define network structure
layers = struct;
layers(1).W = tanhLayer0_W;
layers(1).b = tanhLayer0_b;
%layers(2).W = tanhLayer1_W;
%layers(2).b = tanhLayer1_b;
layers(2).W = outputLayer_W;
layers(2).b = outputLayer_b;

%% Feed-Forward on the train set
train_feat = cell(numLayers+1,1);
train_feat{1} = Train_X2';
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
plot(train_label); 

disp((sum((train_pred-train_label).^2)/length(train_label)))

%% Feed-Forward on the valid set
valid_feat = cell(numLayers+1,1);
valid_feat{1} = Valid_X2';
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
plot(valid_label); 

disp((sum((valid_pred-valid_label).^2)/length(valid_label)))

%% Feed-Forward on the test set
test_feat = cell(numLayers+1,1);
test_feat{1} = Test_X2';
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
plot(test_label); legend({'Prediction-Spy','Real value'});set(gca,'fontsize',20)

disp('Mean squared error (combining signal and spy): ');
disp((sum((test_pred-test_label).^2)/length(test_label)))




