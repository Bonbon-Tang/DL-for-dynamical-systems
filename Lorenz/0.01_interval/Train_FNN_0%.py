import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

step = 1
xyz1 = np.loadtxt('0.0_0.5_0.0_xyz')
input1 = xyz1[:10000-step,:]
for i in range(1,step):
    input_cur = xyz1[i:10000-step+i]
    input1 = np.concatenate((input1,input_cur),axis=1)
output1 = xyz1[step:,:]
# print(input1.shape,output1.shape)

xyz2 = np.loadtxt('1.0_1.0_1.0_xyz')
input2 = xyz2[:10000-step,:]
for i in range(1,step):
    input_cur = xyz2[i:10000-step+i]
    input2 = np.concatenate((input2,input_cur),axis=1)
output2 = xyz2[step:,:]

xyz3 = np.loadtxt('2.0_2.0_2.0_xyz')
input3 = xyz3[:10000-step,:]
for i in range(1,step):
    input_cur = xyz3[i:10000-step+i]
    input3 = np.concatenate((input3,input_cur),axis=1)
output3 = xyz3[step:,:]

train_input = np.concatenate((input1,input2,input3), axis=0)
train_output = np.concatenate((output1,output2,output3), axis=0)
train_input = torch.Tensor(train_input)
train_output = torch.Tensor(train_output)
# print(train_input.shape,train_output.shape)

xyzV = np.loadtxt('0.0_1.5_1.5_xyz')
inputV = xyzV[:10000-step,:]
for i in range(1,step):
    input_cur = xyzV[i:10000-step+i]
    inputV = np.concatenate((inputV,input_cur),axis=1)
outputV = xyzV[step:,:]

val_input = torch.Tensor(inputV)
val_output = torch.Tensor(outputV)

model = nn.Sequential(nn.Linear(3*step,32),
                      nn.ReLU(),
                      nn.Linear(32,3),
                     )

model = torch.load("model_FNN1_0%")

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

# _,_ = train(4001)
# torch.save(model, "model_FNN3_0%")

length = 200
# test_input = torch.Tensor(np.loadtxt('0.2_0.2_0.3_xyz'))[startPoint:startPoint+step,:].reshape(step,3)
# test_result = np.loadtxt('0.2_0.2_0.3_xyz')[startPoint+step:startPoint+step+length,:]
test_input = torch.Tensor(np.loadtxt('0.2_0.2_0.3_xyz'))
test_result = np.loadtxt('0.2_0.2_0.3_xyz')

# def evaluate (initial,length):
#     output = initial
#     for i in range(length):
#         one_output = model(output[-step:].reshape(step*3))
#         output = torch.cat((output,one_output.reshape(1,3)),dim=0)
#
#     return output[-length:,:]
def evaluate (test_input,length):
    output = test_input[:step,:].reshape(step,3)
    for i in range(length-step):
        one_output = model(output[-step:].reshape(step*3))
        output = torch.cat((output,one_output.reshape(1,3)),dim=0)
    output_set = output
    for i in range(1,10000//length):
        output = test_input[length*i:length*i+step, :].reshape(step, 3)
        for i in range(length - step):
            one_output = model(output[-step:].reshape(step * 3))
            output = torch.cat((output, one_output.reshape(1, 3)), dim=0)
        output_set = torch.cat((output_set,output),dim=0)
    return output_set


predict_result = evaluate(test_input,length)
predict_result = predict_result.detach().numpy()

fig,axes = plt.subplots(3,1)
ax1=axes[0]
ax2=axes[1]
ax3=axes[2]

ax1.plot(test_result[:,0],label = "real")
ax1.plot(predict_result[:,0],label = "predict")
ax1.legend()

ax2.plot(test_result[:,1],label = "real")
ax2.plot(predict_result[:,1],label = "predict")
ax2.legend()

ax3.plot(test_result[:,2],label = "real")
ax3.plot(predict_result[:,2],label = "predict")
ax3.legend()

plt.show()

print(np.mean(np.square(predict_result-test_result)))


