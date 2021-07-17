import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# force: (1,300,1)
# state: (1,300,2):(batch,length,(x,v))
# input: (1,300,3):(batch,length,(x,v,F))
step = 1
force1 = np.loadtxt('0.0_0.0_input_10%').reshape(300,1)
state1 = np.loadtxt('0.0_0.0_state_10%').reshape(301,2)
input1 = np.concatenate((state1[:-step,:],force1[:300-step+1,:]),axis = 1)
# print(input1.shape)

force2 = np.loadtxt('0.0_-1.0_input_10%').reshape(300,1)
state2 = np.loadtxt('0.0_-1.0_state_10%').reshape(301,2)
input2 = np.concatenate((state2[:-step,:],force2[:300-step+1,:]),axis = 1)

force3 = np.loadtxt('1.0_1.0_input_10%').reshape(300,1)
state3 = np.loadtxt('1.0_1.0_state_10%').reshape(301,2)
input3 = np.concatenate((state3[:-step,:],force3[:300-step+1,:]),axis = 1)

force4 = np.loadtxt('2.0_1.5_input_10%').reshape(300,1)
state4 = np.loadtxt('2.0_1.5_state_10%').reshape(301,2)
input4 = np.concatenate((state4[:-step,:],force4[:300-step+1,:]),axis = 1)

train_input = np.concatenate((input1,input2,input3,input4), axis=0)
train_output = np.concatenate((state1[step:,:],state2[step:,:],state3[step:,:],state4[step:,:]), axis=0)
train_input = torch.Tensor(train_input)
train_output = torch.Tensor(train_output)
# print(train_input.shape,train_output.shape)

forceV = np.loadtxt('0.5_0.5_input_10%').reshape(300,1)
stateV = np.loadtxt('0.5_0.5_state_10%').reshape(301,2)
inputV = np.concatenate((stateV[:-step,:],forceV[:300-step+1,:]),axis = 1)
outputV = stateV[step:,]
val_input = torch.Tensor(inputV)
val_output = torch.Tensor(outputV)
# print(val_output.shape,val_input.shape)

model = nn.Sequential(nn.Linear(3,20),
                      nn.ReLU(),
                      nn.Linear(20,2),
                     )
model = torch.load("model1")
# for m in model.modules():
#    if isinstance(m,(nn.Linear)):
#         nn.init.kaiming_uniform_(m.weight)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.002)

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
                # torch.save(model, "model_1step_4%")
            if epoch % 200 == 0:
                train_loss.append(loss.tolist())
                val_loss.append(loss_dev.tolist())
                print(f"epoch:{epoch},train_loss:{loss},val_loss:{loss_dev}")
    return train_loss,val_loss


torch.save(model, "model_1step_10%")
# train_loss,val_loss = train(4001)

# testName = '2.0_1.0_'
# testName = '0.0_1.5_'
testName = '0.5_0.5_'

test_initial = torch.Tensor(np.loadtxt(testName+'state')[0,:].reshape(1,2))
test_force = torch.Tensor(np.loadtxt(testName+'input').reshape(300,1))
test_result = np.loadtxt(testName+'state')[1:,:].reshape(300,2)
predict_result = np.zeros(shape=(300,2))

one_result = model(torch.cat((test_initial,test_force[:1,:]),1))
one_result = one_result[:, :2]
predict_result[0,:] = one_result.detach().numpy()

for i in range(1,300):
    one_result = torch.cat((one_result, test_force[i,:].reshape(1,1)),1)
    one_result = model(one_result)
    one_result = one_result[:,:2]
    predict_result[i, :] = one_result.detach().numpy()

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