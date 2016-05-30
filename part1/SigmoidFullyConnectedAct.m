function [act] = SigmoidFullyConnectedAct(input, W, b)
%Compute activation on the layer nodes given input
%input (T,inputSize)
%act (T,outputSize)
%auxData: layer specific auxiliary data

batchSize = size(input,1);
act = input * W + repmat(b,batchSize,1);
act = 1./ (1+exp(-act));

end