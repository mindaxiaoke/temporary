Update May 24, 2016
Version 2
data_cleaning_LR2.m is the new matlab file for cleaning the data, applying linear regression to the signal and spy. The performance is evaluated with correlation value and prediction accuracy (up/down).

dnn_dataClassification.py
The theano script for predicting the up/down of spy using historical signal return and signal return + spy return.

dnn_dataClassification.ipynb
The ipython notebook with the training process of the classification neural network in dnn_dataClassification.py

dnnModelClassification_data1.m
The .m file to load the parameters of the network trained and produce the prediction of the train/valid/test sets. The input feature is the historical signal return.

dnnModelClassification_data3.m
The .m file to load the parameters of the network trained and produce the prediction of the train/valid/test sets. The input feature is the historical signal return and spy return.


Version 1.

Thank you so much for preparing this interesting project. I wish I have more time to further expand the code and method I have in mind to explore more of the data. Here is a brief summary of the code and models trained.

data_cleaning_LR.m
This is the first piece code used for visualizing the data and cleaning the data. It also include a part to apply linear regression to predict future SPY return using:
  1.Past signal return.
  2.Past SPY return.
  3.Combining signal return and SPY return.
This code also saves the data as the input to the DNN.

dnn_Data.py
This is the code for the deep learning model. It trains shallow networks to predict the future SPY return using:
  1.Past SPY return.
  2.Combining signal return and SPY return.
This is to further verify whether the signal adds value to the prediction.

dnn_Data.ipynb
This is the python notebook recording the training of both models in dnn_Data.py

ToTheano.mat
The .mat file storing the training, validation and test set for the dnn_Data.py.

TanhFullyConnected.m
This is a function for the feed-forward network in matlab for using the model trained by Python Theano.

dnnModel_data1.mat
dnnModel_data2.mat
These stores the network architecture trained by the dnn_Data.py. The first one corresponds to using the past SPY return. The second one corresponds to using the combination of signal return and SPY return.

dnnModel_data1.m
dnnModel_data2.m
These codes uses the models trained and stored in the .mat files and feed-forward on the test set to get the prediction values. They also calculate the MSE.