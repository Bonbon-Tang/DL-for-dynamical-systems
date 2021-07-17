import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# force: (1,300,1)
# state: (1,301,2):(batch,length,(x,v))
# input: (1,300,3):(batch,length,(x,v,F))
step = 3
force1 = np.loadtxt('0.0_0.0_input_15%').reshape(300,1)
state1 = np.loadtxt('0.0_0.0_state_15%').reshape(301,2)
input1 = np.concatenate((state1[:301 -step, :], force1[:300 - step + 1, :]), axis=1)
for i in range (1,step):
    input_cur = np.concatenate((state1[i:301-step+i,:],force1[i:300-step+1 +i,:]),axis = 1)
    input1 =  np.concatenate((input1,input_cur),axis= 1)
# print(input1.shape)

force2 = np.loadtxt('0.0_-1.0_input_15%').reshape(300,1)
state2 = np.loadtxt('0.0_-1.0_state_15%').reshape(301,2)
input2 = np.concatenate((state2[:301 -step, :],force2[:300 - step + 1, :]),axis = 1)
for i in range (1,step):
    input_cur = np.concatenate((state2[i:301-step+i,:],force2[i:300-step+1 +i,:]),axis = 1)
    input2 =  np.concatenate((input2,input_cur),axis= 1)

force3 = np.loadtxt('1.0_1.0_input_15%').reshape(300,1)
state3 = np.loadtxt('1.0_1.0_state_15%').reshape(301,2)
input3 = np.concatenate((state3[:301 -step, :],force3[:300 - step + 1, :]),axis = 1)
for i in range (1,step):
    input_cur = np.concatenate((state3[i:301-step+i,:],force3[i:300-step+1 +i,:]),axis = 1)
    input3 =  np.concatenate((input3,input_cur),axis= 1)

force4 = np.loadtxt('2.0_1.5_input_15%').reshape(300,1)
state4 = np.loadtxt('2.0_1.5_state_15%').reshape(301,2)
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
forceV = np.loadtxt('0.0_1.0_input_15%').reshape(300,1)
stateV = np.loadtxt('0.0_1.0_state_15%').reshape(301,2)
inputV = np.concatenate((stateV[:301 -step, :],forceV[:300 - step + 1, :]),axis = 1)
for i in range (1,step):
    input_cur = np.concatenate((stateV[i:301-step+i,:],forceV[i:300-step+1 +i,:]),axis = 1)
    inputV =  np.concatenate((inputV,input_cur),axis= 1)

outputV = stateV[step:,]
val_input = torch.Tensor(inputV)
val_output = torch.Tensor(outputV)
# print(val_output.shape,val_input.shape)
#
model = nn.Sequential(nn.Linear(3*step,20),
                      nn.ReLU(),
                      nn.Linear(20,2),
                     )
#
for m in model.modules():
   if isinstance(m,(nn.Linear)):
        nn.init.kaiming_uniform_(m.weight)

model = torch.load("model_3step_15%")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.004)
#
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
                torch.save(model, "model2")
            if epoch % 200 == 0:
                train_loss.append(loss.tolist())
                val_loss.append(loss_dev.tolist())
                print(f"epoch:{epoch},train_loss:{loss},val_loss:{loss_dev}")
    return train_loss,val_loss

# train_loss,val_loss = train(3001)
# torch.save(model, "model_2step_15%")

testName = '2.0_1.0_'
testName = '0.5_0.5_'
testName = '0.0_1.5_'

test_input = torch.Tensor(np.loadtxt(testName+'state_15%')[:100,:].reshape(100,2))
test_force = torch.Tensor(np.loadtxt(testName+'input_15%')[:150].reshape(150,1))
test_result = np.loadtxt(testName+'state')[100:150,:].reshape(50,2)

def evaluate(input,force,length,step):
    sequence = torch.cat((input,force[:100]),dim=1)
    for i in range (length):
        one_input = sequence[-step:,].reshape(step * 3)
        model.eval()
        one_result = model(one_input)
        one_result = one_result.reshape(1,2)
        one_result =  torch.cat((one_result,force[100+i,].reshape(1,1)),dim = 1)
        sequence = torch.cat((sequence,one_result),dim=0)
    return sequence[-length:,:2]

predict_result = evaluate(test_input,test_force,50,step)
predict_result = predict_result.detach().numpy()

fig,axes = plt.subplots(2,1)
ax1=axes[0]
ax2=axes[1]

ax1.plot(test_result[:,0],label = "real")
ax1.plot(predict_result[:,0],label = "predict")
ax1.legend()

ax2.plot(test_result[:,1],label = "real")
ax2.plot(predict_result[:,1],label = "predict")
ax2.legend()

plt.show()

print(np.mean(np.square(test_result[:,0]-predict_result[:,0])))
print(np.mean(np.square(test_result[:,1]-predict_result[:,1])))