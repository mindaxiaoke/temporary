This README file summarizes the codes for the deep learning model for 1min data.

data_1min.m
The matlab file for cleaning the data and extracting features and targets for training the deep neural networks.

dnn_1minClassification.py
The python theano code for testing deep neural networks with different number of hidden layers/units with early stopping. The parameters of the trained network is saved for latter analysis.

dnnModel_1min_5.m
The matlab file for analyzing the performance of the network and test the cumulative_pnl on the test set.

test_1min_5.ipynb
The ipython notebook with the training process of the deep neural networks with different layers/units.

SoftmaxFullyConnectedAct.m  TanhFullyConnectedAct.m
The matlab files for feed-forward fully connected deep neural networks.

