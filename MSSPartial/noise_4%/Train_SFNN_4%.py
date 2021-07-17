import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

step = 5
hiddenSize = 32
state1 = np.loadtxt('1.0_2.0_state_4%')[:,0].reshape(201,1)
input1 = state1[:201 -step, :]
for i in range (1,step):
    input_cur = state1[i:201-step+i,:]
    input1 =  np.concatenate((input1,input_cur),axis= 1)

state2 = np.loadtxt('2.0_0.5_state_4%')[:,0].reshape(201,1)
input2 = state2[:201 -step, :]
for i in range (1,step):
    input_cur = state2[i:201-step+i,:]
    input2 =  np.concatenate((input2,input_cur),axis= 1)
state1 = np.loadtxt('1.0_2.0_state_4%')[:,0].reshape(201,1)

state3 = np.loadtxt('-2.0_1.0_state_4%')[:,0].reshape(201,1)
input3 = state3[:201 -step, :]
for i in range (1,step):
    input_cur = state3[i:201-step+i,:]
    input3 =  np.concatenate((input3,input_cur),axis= 1)

state4 = np.loadtxt('-1.0_-1.0_state_4%')[:,0].reshape(201,1)
input4 = state4[:201 -step, :]
for i in range (1,step):
    input_cur = state4[i:201-step+i,:]
    input4 =  np.concatenate((input4,input_cur),axis= 1)

train_input = np.concatenate((input1,input2,input3,input4), axis=0)
train_output = np.concatenate((state1[step:,:1],state2[step:,:1],state3[step:,:1],state4[step:,:1]), axis=0)
train_input = torch.Tensor(train_input)
train_output = torch.Tensor(train_output)
# print(train_input.shape,train_output.shape)

stateV = np.loadtxt('0.0_1.0_state_4%')[:,0].reshape(201,1)
inputV = stateV[:201 -step, :]
for i in range (1,step):
    input_cur = stateV[i:201-step+i,:]
    inputV =  np.concatenate((inputV,input_cur),axis= 1)
outputV = stateV[step:,:1]
val_input = torch.Tensor(inputV)
val_output = torch.Tensor(outputV)

model = nn.Sequential(nn.Linear(1*step,hiddenSize),
                      nn.ReLU(),
                      nn.Linear(hiddenSize,1),
                     )

for m in model.modules():
   if isinstance(m,(nn.Linear)):
        nn.init.kaiming_uniform_(m.weight)

# model = torch.load("model1")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

def train(epochs):
    train_loss = []
    val_loss = []
    min_loss = 1e6
    for epoch in range(epochs):
            model.train()
            output = model(train_input)
            loss = criterion(output,train_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            model.eval()
            output_val = model(val_input)
            loss_dev = criterion(output_val,val_output)
            if loss < min_loss:
                min_loss = loss
                torch.save(model, "model1")
            if epoch % 200 == 0:
                train_loss.append(loss.tolist())
                val_loss.append(loss_dev.tolist())
                print(f"epoch:{epoch},train_loss:{loss},val_loss:{loss_dev}")
    return train_loss,val_loss

train_loss,val_loss = train(12001)
# torch.save(model, "model_FNN8_32_4%")

testNames = ['0.5_0.5_','-0.5_0.5_','1.0_0.0_']
result = []
for testName in testNames:

    test_state = torch.Tensor(np.loadtxt(testName+'state_4%').reshape(201,2))
    test_input = torch.Tensor(np.loadtxt(testName+'state_4%')[:step,0].reshape(step,1))
    test_result = np.loadtxt(testName+'state_4%')[step:,0].reshape(201-step,1)

    def evaluate(initial,length):
        result = initial
        for i in range(length):
            input = result[-step:,:].reshape(1,step)
            one_output = model(input)
            one_output = one_output.reshape(1,1)
            result = torch.cat((result,one_output),dim=0)

        return result[-length:,:]

    predict_result = evaluate(test_input,201-step)
    predict_result = predict_result.detach().numpy()

    fig,axes = plt.subplots(2,1)
    ax1=axes[0]

    ax1.plot(test_result[:,0],label = "real")
    ax1.plot(predict_result[:,0],label = "predict")
    ax1.legend()

    plt.show()

    result.append(np.mean(np.square(test_result-predict_result)))

print(result)
