# DL for dynamical systems


Previously, many works has been showing the possibility of using Neural Networks to model dynamical systems. However, 
seldom of them considered different conditions of dynamical systems. The most important features of dynamical systems
might be the noise level, the measurement interval and the target of modelling. Therefore, the purpose of this project
is to show how these features change the choice of deep learning methods, such as FNN, RNNs and Transformer.

#######################################################

In the first experiment, the used dynaimical system is the fully observed mass spring system. 8 sets of data (301x2) was 
generated via given mathematical equations.There are  three different targets within this experiment, namely the Long-Time
Prediction, Short-Time Prediction and the interval-required Prediction. 
