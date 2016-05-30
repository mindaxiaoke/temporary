%% DNN Test
%create a Deep neural network based on the passed parameters from Python
%theano
clear; close; clc
load ToTheanoClassification.mat
load dnnModel_data_Class_Train3_save1.mat
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
train_feat{1} = Train_X3';
train_label = Train_Y;
for i=1:numLayers
    if (i<numLayers)
        [train_feat{i+1}] = TanhFullyConnectedAct(train_feat{i}, layers(i).W, layers(i).b);
    else
        [train_feat{i+1}] = SoftmaxFullyConnectedAct(train_feat{i}, layers(i).W, layers(i).b);        
    end
end
train_pred = train_feat{numLayers+1};
[~,train_pred_label] = max(train_pred,[],2);
train_pred_label = train_pred_label;
figure('color','w')
stem(train_pred_label);hold on;
stem(train_label,'r'); 

disp('Classification accuracy')
disp(sum(train_pred_label==train_label')/length(train_label))

%% Feed-Forward on the valid set
valid_feat = cell(numLayers+1,1);
valid_feat{1} = Valid_X3';
valid_label = Valid_Y;
for i=1:numLayers
    if (i<numLayers)
        [valid_feat{i+1}] = TanhFullyConnectedAct(valid_feat{i}, layers(i).W, layers(i).b);
    else
        [valid_feat{i+1}] = SoftmaxFullyConnectedAct(valid_feat{i}, layers(i).W, layers(i).b);        
    end
end
valid_pred = valid_feat{numLayers+1};
[~,valid_pred_label] = max(valid_pred,[],2);
valid_pred_label = valid_pred_label;
figure('color','w')
stem(valid_pred_label);hold on;
stem(valid_label,'r'); 

disp('Classification accuracy')
disp(sum(valid_pred_label==valid_label')/length(valid_label))

%% Feed-Forward on the test set
test_feat = cell(numLayers+1,1);
test_feat{1} = Test_X3';
test_label = Test_Y;
for i=1:numLayers
    if (i<numLayers)
        [test_feat{i+1}] = TanhFullyConnectedAct(test_feat{i}, layers(i).W, layers(i).b);
    else
        [test_feat{i+1}] = SoftmaxFullyConnectedAct(test_feat{i}, layers(i).W, layers(i).b);        
    end
end
test_pred = test_feat{numLayers+1};
[~,test_pred_label] = max(test_pred,[],2);
test_pred_label = test_pred_label;
figure('color','w')
stem(test_pred_label);hold on;
stem(test_label,'r'); legend({'Prediction-Signal','Real value'});set(gca,'fontsize',20)

disp('Classification accuracy')
disp(sum(test_pred_label==test_label')/length(test_label))

disp('Correlation in test set')
disp(corrcoef(test_pred_label,double(test_label)))

