function [act] = ReluFullyConnectedAct(input, W, b)
%Compute activation on the layer nodes given input
%input (T,inputSize)
%act (T,outputSize)
%auxData: layer specific auxiliary data

batchSize = size(input,1);
act = input * W + repmat(b,batchSize,1);
act = max(0,act);

end