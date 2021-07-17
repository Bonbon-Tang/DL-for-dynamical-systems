import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# force: (1,300,1)
# state: (1,301,2):(batch,length,(x,v))
# input: (1,300,3):(batch,length,(x,v,F))
step = 5
force1 = np.loadtxt('0.0_0.0_input_10%').reshape(300,1)
state1 = np.loadtxt('0.0_0.0_state_10%').reshape(301,2)
input1 = np.concatenate((state1[:301 -step, :], force1[:300 - step + 1, :]), axis=1)
for i in range (1,step):
    input_cur = np.concatenate((state1[i:301-step+i,:],force1[i:300-step+1 +i,:]),axis = 1)
    input1 =  np.concatenate((input1,input_cur),axis= 1)
# print(input1.shape)

force2 = np.loadtxt('0.0_-1.0_input_10%').reshape(300,1)
state2 = np.loadtxt('0.0_-1.0_state_10%').reshape(301,2)
input2 = np.concatenate((state2[:301 -step, :],force2[:300 - step + 1, :]),axis = 1)
for i in range (1,step):
    input_cur = np.concatenate((state2[i:301-step+i,:],force2[i:300-step+1 +i,:]),axis = 1)
    input2 =  np.concatenate((input2,input_cur),axis= 1)

force3 = np.loadtxt('1.0_1.0_input_10%').reshape(300,1)
state3 = np.loadtxt('1.0_1.0_state_10%').reshape(301,2)
input3 = np.concatenate((state3[:301 -step, :],force3[:300 - step + 1, :]),axis = 1)
for i in range (1,step):
    input_cur = np.concatenate((state3[i:301-step+i,:],force3[i:300-step+1 +i,:]),axis = 1)
    input3 =  np.concatenate((input3,input_cur),axis= 1)

force4 = np.loadtxt('2.0_1.5_input_10%').reshape(300,1)
state4 = np.loadtxt('2.0_1.5_state_10%').reshape(301,2)
input4 = np.concatenate((state4[:301 -step, :],force4[:300 - step + 1, :]),axis = 1)
for i in range (1,step):
    input_cur = np.concatenate((state4[i:301-step+i,:],force4[i:300-step+1 +i,:]),axis = 1)
    input4 =  np.concatenate((input4,input_cur),axis= 1)
#
train_input = np.concatenate((input1,input2,input3,input4), axis=0)
train_output = np.concatenate((state1[step:,:],state2[step:,:],state3[step:,:],state4[step:,:]), axis=0)
train_input = torch.Tensor(train_input)
train_output = torch.Tensor(train_output)
# print(train_input.shape,train_output.shape)
#
forceV = np.loadtxt('1.0_0.0_input_10%').reshape(300,1)
stateV = np.loadtxt('1.0_0.0_state_10%').reshape(301,2)
inputV = np.concatenate((stateV[:301 -step, :],forceV[:300 - step + 1, :]),axis = 1)
for i in range (1,step):
    input_cur = np.concatenate((stateV[i:301-step+i,:],forceV[i:300-step+1 +i,:]),axis = 1)
    inputV =  np.concatenate((inputV,input_cur),axis= 1)

outputV = stateV[step:,]
val_input = torch.Tensor(inputV)
val_output = torch.Tensor(outputV)
# print(val_output.shape,val_input.shape)
#
@variational_estimator
class BayesianRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim,output_dim):
        super().__init__()
        # self.linear = nn.Linear(input_dim, output_dim)
        self.blinear1 = BayesianLinear(input_dim, hidden_dim)
        self.blinear2 = BayesianLinear(hidden_dim, output_dim)

    def forward(self, x):
        x_ = self.blinear1(x)
        x_ = F.relu(x_)
        return self.blinear2(x_)

BFNN = BayesianRegressor(input_dim=3*step,hidden_dim=32,output_dim=2)
# BFNN = torch.load("BayesianFNN5")

optimizer = torch.optim.Adam(BFNN.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def evaluate(input,force,length,step):
    sequence = torch.cat((input,force[:100]),dim=1)
    for i in range (length):
        one_input = sequence[-step:,].reshape(step * 3)
        BFNN.eval()
        one_result = BFNN(one_input)
        one_result = one_result.reshape(1,2)
        one_result =  torch.cat((one_result,force[100+i,].reshape(1,1)),dim = 1)
        sequence = torch.cat((sequence,one_result),dim=0)

    return sequence[-length:,:2]

def GetInterval(input,force,length,step,samples):
    output_set = evaluate(input,force,length,step)
    output_set = output_set.reshape(1,length,2)
    for i in range(samples-1):
        output = evaluate(input,force,length,step)
        output = output.reshape(1,length,2)
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

def train(epochs):
    min_loss = 1e6
    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = BFNN.sample_elbo(inputs=train_input,
                                 labels=train_output,
                                 criterion=criterion,
                                 sample_nbr=3,
                                 complexity_cost_weight=1e-5)
        loss.backward()
        optimizer.step()

        if loss < min_loss:
            min_loss = loss
            torch.save(BFNN, "model1")

        if epoch % 200 == 0:
            train_loss.append(loss.tolist())
            BFNN.eval()
            output_val = BFNN(val_input)
            loss_dev = criterion(output_val, val_output)
            val_loss.append(loss_dev.tolist())
            print(f"epoch:{epoch},train_loss:{loss},val_loss:{loss_dev}")
    return

train(2001)

# torch.save(BFNN,"BayesianFNN5")

def GetAccuracy(upper,lower,data):
    total = data.shape[0]
    accurate = 0
    for i in range(total):
        if data[i] <= upper[i] and data[i] >= lower[i]:
            accurate += 1

    return accurate/total

# testNames = ['2.0_1.0_','0.5_0.5_','0.0_1.5_']
testName = '2.0_1.0_'
# testName = '0.5_0.5_'
# testName = '0.0_1.5_'

length = 50

test_input = torch.Tensor(np.loadtxt(testName+'state')[:100,:].reshape(100,2))
test_force = torch.Tensor(np.loadtxt(testName+'input')[:100+length].reshape(100+length,1))
test_output = np.loadtxt(testName+'state')[100:100+length,:].reshape(length,2)
x_real = test_output[ :, 0]
v_real = test_output[ :, 1]

x_predict,x_std,v_predict,v_std = GetInterval(test_input,test_force,length,step,100)

multiplier = 2

print(GetAccuracy(x_predict + x_std* multiplier,x_predict - x_std * multiplier,x_real))
print(GetAccuracy(v_predict + v_std* multiplier,v_predict - v_std * multiplier,v_real))
print(np.mean(x_std),np.mean(v_std))
print(np.mean(np.abs(x_predict)),np.mean(np.abs(v_predict)))

fig,axes = plt.subplots(2,1)
ax1=axes[0]
ax2=axes[1]

step = np.linspace(1,length,length).reshape(length)
ax1.plot(step,x_real,label = "real")
ax1.plot(step,x_predict,label = "predict")
ax1.fill_between(step,x_predict - x_std * multiplier, x_predict + x_std * multiplier, alpha=0.2)
ax1.legend()

ax2.plot(step,v_real,label = "real")
ax2.plot(step,v_predict,label = "predict")
ax2.fill_between(step,v_predict - v_std * multiplier, v_predict + v_std * multiplier, alpha=0.2)
ax2.legend()

plt.show()

