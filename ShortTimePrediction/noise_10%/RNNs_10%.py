import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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
forceV = np.loadtxt('1.0_0.0_input_10%').reshape(1, 300, 1)
stateV = np.loadtxt('1.0_0.0_state_10%').reshape(1, 301, 2)
val_input = np.concatenate((stateV[:,:100,:],forceV[:,:100,:]),axis = 2)
val_force = forceV[:,100:150,:]
val_output = stateV[:,100:150,:]
val_input = torch.Tensor(val_input)
val_force = torch.Tensor(val_force)
val_output = torch.Tensor(val_output)
# print(val_output.shape,val_input.shape,val_force.shape)
#
class GRU(nn.Module):
    def __init__(self, input_size = 3,hidden_size = 32, output_size = 2, n_layers=1):
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.n_layers = n_layers

        #encoder and decoder have different parameters
        self.gru = nn.LSTM(input_size, hidden_size, n_layers,batch_first=True)
        # self.gru = nn.GRU(input_size, hidden_size, n_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input:torch.Tensor, train = False, force = None, predict_length = 50):
        hidden = None
        if train == True:
            encoder_outputs, hidden = self.gru(input, hidden)

            output = self.out(encoder_outputs)

            return output
        else:
            encoder_outputs, hidden = self.gru(input, hidden)

            decoder_input = encoder_outputs[:, -1, :].reshape(-1, 1, self.hidden_size)
            decoder_input = self.out(decoder_input)

            decoder_output_set = torch.zeros([0])
            decoder_output_set = torch.cat((decoder_output_set, decoder_input), dim=1)

            for i in range (predict_length-1):
                one_force = force[:,i,:].reshape(-1,1,1)
                decoder_input = torch.cat((decoder_input,one_force),dim=2)
                one_output, hidden = self.gru(decoder_input, hidden)
                decoder_output = self.out(one_output)
                decoder_output_set = torch.cat((decoder_output_set, decoder_output), dim=1)
                decoder_input = decoder_output
            return decoder_output_set
#
GRU_model = GRU()
GRU_model = torch.load("GRU_32_model_10%")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(GRU_model.parameters(),lr=0.01)

def train(epochs):
    train_loss = []
    # minimum = 1e6
    for epoch in range(epochs):
        GRU_model.train()
        output = GRU_model(input = train_input,train = True)
        loss = criterion(output,train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 20 == 0:
            GRU_model.eval()
            val_predict = GRU_model(input=val_input, force=val_force,train = False)
            val_loss = criterion(val_predict,val_output)
            train_loss.append(loss.tolist())
            print(f"epoch:{epoch},train_loss:{loss},val_loss:{val_loss}")
        torch.save(GRU_model, "model2")

    return train_loss


# torch.save(GRU_model, "GRU_32_model_10%")
# torch.save(GRU_model, "LSTM_32_model_10%")

# trainLoss = train(801)

testName = '2.0_1.0_'
# testName = '0.5_0.5_'
# testName = '0.0_1.5_'

test_force = np.loadtxt(testName+'input_10%').reshape(1, 300, 1)
test_state = np.loadtxt(testName+'state_10%').reshape(1, 301, 2)
test_output = np.loadtxt(testName+'state').reshape(1, 301, 2)[:,100:150,:]
test_input = np.concatenate((test_state[:,:100,:],test_force[:,:100,:]),axis = 2)
test_force = torch.Tensor(test_force[:,100:150,:])
test_state = torch.Tensor(test_state)
test_input = torch.Tensor(test_input)

output = GRU_model(input=test_input,force =test_force,train = False,predict_length = 50)

output = output.detach().numpy()
x_predict = output[0,:,0]
v_predict = output[0,:,1]

val_output = val_output.numpy()
x_real = test_output[0,:,0]
v_real = test_output[0,:,1]

fig,axes = plt.subplots(2,1)
ax1=axes[0]
ax2=axes[1]

ax1.plot(x_real,label = "real")
ax1.plot(x_predict,label = "predict")
ax1.legend()

ax2.plot(v_real,label = "real")
ax2.plot(v_predict,label = "predict")
ax2.legend()

plt.show()

print(np.mean(np.square(x_predict-x_real)))
print(np.mean(np.square(v_predict-v_real)))

