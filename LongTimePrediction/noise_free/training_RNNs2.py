import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# target:using first 50 steps to predict next 250 steps
input_length = 100

force1 = np.loadtxt('0.0_0.0_input_sin').reshape(1,300,1)
state1 = np.loadtxt('0.0_0.0_state')[0,:].reshape(1,301,1)
input1 = np.concatenate((state1[:,:input_length,:],force1[:,:input_length,:]),axis = 2)
input_force1 = force1[:,input_length:,:]
output1 = state1[:,input_length+1:,:]

force2 = np.loadtxt('2.0_1.0_input_cos').reshape(1,300,1)
state2 = np.loadtxt('2.0_1.0_state')[0,:].reshape(1,301,1)
input2 = np.concatenate((state2[:,:input_length,:],force2[:,:input_length,:]),axis = 2)
input_force2 = force2[:,input_length:,:]
output2 = state2[:,input_length+1:,:]

force3 = np.loadtxt('2.1_1.0_input_bang').reshape(1,300,1)
state3 = np.loadtxt('2.1_1.0_state')[0,:].reshape(1,301,1)
input3 = np.concatenate((state3[:,:input_length,:],force3[:,:input_length,:]),axis = 2)
input_force3 = force3[:,input_length:,:]
output3 = state3[:,input_length+1:,:]

force4 = np.loadtxt('1.0_1.0_input_0.3s').reshape(1,300,1)
state4 = np.loadtxt('1.0_1.0_state')[0,:].reshape(1,301,1)
input4 = np.concatenate((state4[:,:input_length,:],force4[:,:input_length,:]),axis = 2)
input_force4 = force4[:,input_length:,:]
output4 = state4[:,input_length+1:,:]

train_X = np.concatenate((input1,input2,input3,input4), axis=0)
train_Y = np.concatenate((output1,output2,output3,output4), axis=0)
train_F = np.concatenate((input_force1,input_force2,input_force3,input_force4), axis=0)
train_X = torch.Tensor(train_X)
train_Y = torch.Tensor(train_Y)
train_F = torch.Tensor(train_F)
# print(train_X.shape,train_Y.shape,train_F.shape)

forceV = np.loadtxt('0.5_1.0_input_bang').reshape(1,300,1)
stateV = np.loadtxt('0.5_1.0_state')[0,:].reshape(1,301,1)
inputV = np.concatenate((stateV[:,:input_length,:],forceV[:,:input_length,:]),axis = 2)
input_forceV = forceV[:,input_length:,:]
outputV = stateV[:,input_length+1:,:]

val_X = torch.Tensor(inputV)
val_F = torch.Tensor(input_forceV)
val_Y = torch.Tensor(outputV)
#
class GRU(nn.Module):
    def __init__(self, encoder_input = 2,decoder_input = 1,hidden_size = 16, output_size = 1, n_layers=1):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.n_layers = n_layers

        #encoder and decoder have different parameters
        self.encoder = nn.GRU(encoder_input, hidden_size, n_layers,batch_first=True)
        self.decoder = nn.GRU(decoder_input, hidden_size, n_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input:torch.Tensor, force:torch.Tensor):
        hidden = None
        encoder_outputs, hidden = self.encoder(input,hidden)

        decoder_outputs,hidden = self.decoder(force,hidden)

        model_outputs = self.out(decoder_outputs)
        return model_outputs

model_GRU = GRU()
GRU_model = torch.load("model2")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_GRU.parameters(),lr=0.01)
#
def train(epochs):
    train_loss = []
    # minimum = 1e6
    for epoch in range(epochs):
        model_GRU.train()
        predict_output = model_GRU(input = train_X,force = train_F)
        loss = criterion(predict_output,train_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 20 == 0:
            model_GRU.eval()
            val_predict = model_GRU(input=val_X, force=val_F)
            val_loss = criterion(val_predict,val_Y)
            train_loss.append(loss.tolist())
            print(f"epoch:{epoch},train_loss:{loss},val_loss:{val_loss}")
        torch.save(model_GRU, "model3")

    return train_loss
#
trainLoss = train(400)
#
output = model_GRU(input=val_X,force =val_F)
output = output.detach().numpy().reshape(-1)
val_Y = val_Y.numpy().reshape(-1)

plt.plot(val_Y)
plt.plot(output)
plt.show()
