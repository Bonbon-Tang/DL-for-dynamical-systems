# DL for dynamical systems


Previously, many works has been showing the possibility of using Neural Networks to model dynamical systems. However, 
seldom of them considered different conditions of dynamical systems. The most important features of dynamical systems
might be the noise level, the measurement interval and the target of modelling. Therefore, the purpose of this project
is to show how these features change the choice of deep learning methods, such as FNN, RNNs and Transformer.

#######################################################

In the first experiment, the used dynaimical system is the fully observed mass spring system. 8 sets of data (301x2) was 
generated via given mathematical equations.There are  three different targets within this experiment, namely the Long-Time
Prediction, Short-Time Prediction and the interval-required Prediction. 

For the **Long-Time Prediction**, the target is to get the general trend of dynamics given the ground-truth initial value. Models
involved were basic FNN, GRU, LSTM, Transformer and Transformer with limited sight(5 previous steps). The general result from
this experiment shows that with the increase of the noise level, more information (length of step needed to sent to the neural 
network input); however, too much information will also lead to deviation, such as the full-sight Transformer. For noise Free 
model, the basic FNN performed the best, and then its performance decreased very quickly. GRU and LSTM has not much difference 
in terms of the prediction accuracy. They had seemingly information between 2-3 steps and performed extremely well for noise 
under 20%. For high-noise system, Transformer with 5 step sight performed the best. This might be because it has the most 
information among all models. More detailed result could be seen in the original paper or the excel file under the "Long-Time-
Prediction" directory. All trained model with different noise could also be seen in the "Long-Time-Prediction" directory.

For the **short-Time Prediction**, the target is to give prediction for next steps(here is 50 steps) based on previous noise-have 
information. Model invovled were FNN with 1,2,3 step/steps information, GRU, LSTM and Transformer with sight of 3,5,10. The result
was quite similar to those in the Long-Time Prediction. FNN with 1 step information performed the best while GRU and LSTM performed
the best when the noise was around 10%. Then FNN with 3 steps information outperformed any other model when the noise was beteen 10%
and 20%. Transformer with 5 steps sight was the best when the noise was over 20% while Transformer with 10 steps information seems 
too much and was harmful for modelling. More detailed result could be seen in the original paper or the excel file under the "short-Time-
Prediction" directory. All trained model with different noise could also be seen in the "short-Time-Prediction" directory.

Based on the results, the needed information (how much steps to the input) seems having large correlation to the noise level.
When the noise-level is constant, there might be a best information sent to the Neural Network input.

For the **interval-required modelling**, the target is to give confidence interval for prediction. The model is assumed to have 10 % of noise
The evaluation was based on how manypoints are within the confidence interval. Model involved here were Bayesian FNN and Bayesian LSTM. 
The results was that Bayesian FNN performed better than Bayesian LSTM in terms of the balance between confidence interval and the accuracy.
Increasing the step information in here seems has limited inference. However, both models could not achieve perfect fitting. In other word,
achieving 90% accuracy when the confidence interval was twice larger than 10%; Nevertherless, the result was quite close to 90%. If the predicted
steps reduced from 50 steps to 20 or 30 steps, the perfect accuracy might be achieved. 

#######################################################

The second experiment was also based on the mass spring system; however, the observed variable is only the position while the velocity is unkown.
This might also be a condition in dynamical systems, but hte un-observed variables should be related to those observed; otherwise, modelling is
by no means possible. The only worked model here is the FNN with different input-length. With the increase of the noise, more step information is
needed. However, if too much information was given, the training is likely to overfit and deviate. Detailed reuslt could also be seen in excel file
under the directory "MSSpartial" or the original paper. 

#######################################################

The third experiment was the Lorenz system, a chaotic model. The target is to show whether DL models could capture the dynamics ot the chaotic models.
The interval was 0.01s. The model involved was FNN with different length of input, Transformer and LSTM. The evaluation was the MSE loss of prediction 
given every 20,50,100,200 ground-truth points. If the input was only 1 point, FNN performed the best, similar to the result in Long-time Prediction 
and short-time Prediction for noise-free modelling. However, when the input could be all previous points, LSTM performed even better than the FNN model. 
Both these two models could generally capture the dynamics for 200 point prediction, usually greater than a cycle of changing the "steady side". 

The fouth experiment was also the Lorenz system but with changed measurement interval. The measurement interval was 0.01,0.02,0.05,0.10s in this case.
The model involved was FNN and LSTM, the best 2 in the previous experiment. With the increase of measurement interval, the model could first given more
accurate prediction, but then both two models performed less accuracy. It seems that there is a best measurement interval for each model to balance the
affect of catching the dynamics and reducing the error accumulation. 


