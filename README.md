# DL for dynamical systems


Previously, many works has been showing the possibility of using Neural Networks to model dynamical systems. However, 
seldom of them considered different conditions of dynamical systems. The most important features of dynamical systems
might be the noise level, the measurement interval and the target of modelling. Therefore, the purpose of this project
is to show how these features change the choice of deep learning methods, such as FNN, RNNs and Transformer.

#######################################################

In the first experiment, the used dynaimical system is the fully observed mass spring system. 8 sets of data (301x2) was 
generated via given mathematical equations.There are  three different targets within this experiment, namely the Long-Time
Prediction, Short-Time Prediction and the interval-required Prediction. 

For the Long-Time Prediction, the target is to get the general trend of dynamics given the ground-truth initial value. Models
involved were basic FNN, GRU, LSTM, Transformer and Transformer with limited sight(5 previous steps). The general result from
this experiment shows that with the increase of the noise level, more information (length of step needed to sent to the neural 
network input); however, too much information will also lead to deviation, such as the full-sight Transformer. For noise Free 
model, the basic FNN performed the best, and then its performance decreased very quickly. GRU and LSTM has not much difference 
in terms of the prediction accuracy. They had seemingly information between 2-3 steps and performed extremely well for noise 
under 20%. For high-noise system, Transformer with 5 step sight performed the best. This might be because it has the most 
information among all models.


