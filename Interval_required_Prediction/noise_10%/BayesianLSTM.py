import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

observed = 100
# force: (1,300,1)
# state: (1,300,2):(batch,length,(x,v))
# input: (1,300,3):(batch,length,(x,v,F))
force1 = np.loadtxt('0.0_0.0_input_10%').reshape(1,300,1)
state1 = np.loadtxt('0.0_0.0_state_10%').reshape(1,301,2)
input1 = np.concatenate((state1[:,:-1,:],force1),axis = 2)

# print(input1.shape)

force2 = np.loadtxt('0.0_-1.0_input_10%').reshape(1,300,1)
state2 = np.loadtxt('0.0_-1.0_state_10%').reshape(1,301,2)
input2 = np.concatenate((state2[:,:-1,:],force2),axis = 2)
#
force3 = np.loadtxt('1.0_1.0_input_10%').reshape(1,300,1)
state3 = np.loadtxt('1.0_1.0_state_10%').reshape(1,301,2)
input3 = np.concatenate((state3[:,:-1,:],force3),axis = 2)
#
force4 = np.loadtxt('2.0_1.5_input_10%').reshape(1,300,1)
state4 = np.loadtxt('2.0_1.5_state_10%').reshape(1,301,2)
input4 = np.concatenate((state4[:,:-1,:],force4),axis = 2)
#
train_input = np.concatenate((input1,input2,input3,input4), axis=0)
train_output = np.concatenate((state1[:,1:,:],state2[:,1:,:],state3[:,1:,:],state4[:,1:,:]), axis=0)
train_input = torch.Tensor(train_input)
train_output = torch.Tensor(train_output)
# print(train_input.shape,train_output.shape)
#
forceV = np.loadtxt('1.0_0.0_input_10%').reshape(1,300,1)
stateV = np.loadtxt('1.0_0.0_state_10%').reshape(1,301,2)
inputV = np.concatenate((state4[:,:-1,:],force4),axis = 2)
val_input = torch.Tensor(inputV)
val_output = torch.Tensor(stateV[:,1:,:])

TestName = "2.0_1.0_"
# TestName = "0.5_0.5_"
# TestName = "0.0_1.5_"


# forceT = np.loadtxt(TestName + 'input_10%').reshape(1, 300, 1)
# stateT = np.loadtxt(TestName + 'state_10%').reshape(1, 301, 2)
# test_input = np.concatenate((stateT[:, :100, :], forceT[:, :100, :]), axis = 2)
# test_force = forceT[:, 100:150, :]
# test_output = np.loadtxt(TestName + 'state').reshape(1, 301, 2)[:, 100:150, :]
# test_input = torch.Tensor(test_input)
# test_force = torch.Tensor(test_force)
# # val_output = torch.Tensor(val_output)
# print(val_output.shape,val_input.shape,val_force.shape)
#
@variational_estimator
class BNN_model(nn.Module):
    def __init__(self,input_size,hidden_size,output_size):
        super(BNN_model, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size

        self.BNN = BayesianLSTM(input_size, hidden_size, prior_sigma_1= 0.1, prior_pi= 0.1, posterior_rho_init=-3.0)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self,input,predict_length=0, force = None):
        #force has same length as the predicted length
        if predict_length == 0:
            lstm_output, _ = self.BNN(input)
            output = self.out(lstm_output)
            return output

        else:
            encoder_outputs, hidden = self.BNN(input)

            decoder_input = encoder_outputs[:, -1, :].reshape(-1, 1, self.hidden_size)
            decoder_input = self.out(decoder_input)

            decoder_output_set = torch.zeros([0])
            decoder_output_set = torch.cat((decoder_output_set, decoder_input), dim=1)
            #shape(1,1,2)

            for i in range(predict_length - 1):
                one_force = force[:, i, :].reshape(-1, 1, 1)
                decoder_input = torch.cat((decoder_input, one_force), dim=2)
                one_output, hidden = self.BNN(decoder_input, hidden)
                decoder_output = self.out(one_output)
                decoder_output_set = torch.cat((decoder_output_set, decoder_output), dim=1)
                decoder_input = decoder_output
                #outputshape:(1,predict,2)
            return decoder_output_set

BayesianNN = BNN_model(3,32,2)

BayesianNN = torch.load("BayesianLSTM")
criterion = nn.MSELoss()
optimizer = optim.Adam(BayesianNN.parameters(), lr=0.01)

def train(epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = BayesianNN.sample_elbo(inputs=train_input,
                               labels=train_output,
                               criterion=criterion,
                               sample_nbr=3,
                               complexity_cost_weight = 1e-8)
        loss.backward()
        optimizer.step()
        if epoch % 20==0:
            print("Iteration: {} loss: {:.4f}".format(str(epoch), loss))
    torch.save(BayesianNN,"model1")
    return

# train(401)

Length = 50


forceT = np.loadtxt(TestName + 'input_10%').reshape(1, 300, 1)
stateT = np.loadtxt(TestName + 'state_10%').reshape(1, 301, 2)
test_input = np.concatenate((stateT[:, :100, :], forceT[:, :100, :]), axis = 2)
test_force = forceT[:, 100:100+Length, :]
test_output = np.loadtxt(TestName + 'state').reshape(1, 301, 2)[:, 100:100+Length, :]
test_input = torch.Tensor(test_input)
test_force = torch.Tensor(test_force)

x_real = test_output[0, :, 0]
v_real = test_output[0, :, 1]

def evaluate(samples):
    output_set = BayesianNN(input=test_input, predict_length = Length, force = test_force)
    for i in range(samples-1):
        output = BayesianNN(input=test_input, predict_length = Length, force = test_force)
        output_set = torch.cat((output_set,output),dim=0)
    x_mean = torch.mean(output_set[:,:,0],dim=0)
    x_std = torch.std(output_set[:,:,0],dim=0)
    v_mean = torch.mean(output_set[:, :, 1],dim=0)
    v_std = torch.std(output_set[:, :, 1],dim=0)
    x_mean = x_mean.detach().numpy()
    x_std = x_std.detach().numpy()
    v_mean = v_mean.detach().numpy()
    v_std = v_std.detach().numpy()
    return x_mean,x_std,v_mean,v_std

def GetAccuracy(upper,lower,data):
    total = Length
    accurate = 0
    for i in range(total):
        if data[i] <= upper[i] and data[i] >= lower[i]:
            accurate += 1

    return accurate/total


# torch.save(BayesianNN,"BayesianLSTM")


x_predict,x_std,v_predict,v_std = evaluate(100)


multiplier = 2

print(GetAccuracy(x_predict + x_std* multiplier,x_predict - x_std * multiplier,x_real))
print(GetAccuracy(v_predict + v_std* multiplier,v_predict - v_std * multiplier,v_real))
print(np.mean(x_std),np.mean(v_std))
print(np.mean(np.abs(x_predict)),np.mean(np.abs(v_predict)))

fig,axes = plt.subplots(2,1)
ax1=axes[0]
ax2=axes[1]

step = np.linspace(1,Length,Length).reshape(Length)
ax1.plot(step,x_real,label = "real")
ax1.plot(step,x_predict,label = "predict")
ax1.fill_between(step,x_predict - x_std * multiplier, x_predict + x_std * multiplier, alpha=0.2)
ax1.legend()

ax2.plot(step,v_real,label = "real")
ax2.plot(step,v_predict,label = "predict")
ax2.fill_between(step,v_predict - v_std * multiplier, v_predict + v_std * multiplier, alpha=0.2)
ax2.legend()

plt.show()




