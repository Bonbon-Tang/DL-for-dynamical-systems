import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
#this model does not work


step = 2
state1 = np.loadtxt('1.0_2.0_state')[:,0].reshape(1,201,1)
input1 = state1[:,:201 -step, :]
for i in range (1,step):
    input_cur = state1[:,i:201-step+i,:]
    input1 =  np.concatenate((input1,input_cur),axis= 2)

state2 = np.loadtxt('2.0_0.5_state')[:,0].reshape(1,201,1)
input2 = state2[:,:201 -step, :]
for i in range (1,step):
    input_cur = state2[:,i:201-step+i,:]
    input2 =  np.concatenate((input2,input_cur),axis= 2)

state3 = np.loadtxt('-2.0_1.0_state')[:,0].reshape(1,201,1)
input3 = state3[:,:201 -step, :]
for i in range (1,step):
    input_cur = state3[:,i:201-step+i,:]
    input3 =  np.concatenate((input3,input_cur),axis= 2)

state4 = np.loadtxt('-1.0_-1.0_state')[:,0].reshape(1,201,1)
input4 = state4[:,:201 -step, :]
for i in range (1,step):
    input_cur = state4[:,i:201-step+i,:]
    input4 =  np.concatenate((input4,input_cur),axis= 2)

train_input = np.concatenate((input1,input2,input3,input4), axis=0)
train_output = np.concatenate((state1[:,step:,:1],state2[:,step:,:1],state3[:,step:,:1],state4[:,step:,:1]), axis=0)
train_input = torch.Tensor(train_input)
train_output = torch.Tensor(train_output)
# print(train_input.shape,train_output.shape)

stateV = np.loadtxt('0.0_1.0_state')[:,0].reshape(1,201,1)
inputV = stateV[:,:201 -step, :]
for i in range (1,step):
    input_cur = stateV[:,i:201-step+i,:]
    inputV =  np.concatenate((inputV,input_cur),axis= 2)
outputV = stateV[:,step:,:1]
val_input = torch.Tensor(inputV)
val_output = torch.Tensor(outputV).reshape(-1)

class RNN(nn.Module):
    def __init__(self, input_size = 1*step,hidden_size = 48, output_size = 1, n_layers=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.n_layers = n_layers

        #encoder and decoder have different parameters
        self.rnns = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        # self.gru = nn.GRU(input_size, hidden_size, n_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input:torch.Tensor,predict_length = 0):
        hidden = None
        if predict_length == 0:
            encoder_outputs, hidden = self.rnns(input, hidden)

            output = self.out(encoder_outputs)

            return output
        else:
            total_sequence = input.reshape(-1)

            encoder_outputs, hidden = self.rnns(input, hidden)

            decoder_input = encoder_outputs[:, -1, :].reshape(-1, 1, self.hidden_size)
            decoder_input = self.out(decoder_input)

            total_sequence = torch.cat((total_sequence,decoder_input.reshape(-1)))

            for i in range (predict_length-1):
                one_input = total_sequence[-step:].reshape(-1,1,step)
                one_output, hidden = self.rnns(one_input, hidden)
                one_output = self.out(one_output)
                total_sequence = torch.cat((total_sequence, one_output.reshape(-1)))


            return total_sequence[-predict_length:]

LSTM_model = RNN()
LSTM_model = torch.load("model2")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(LSTM_model.parameters(),lr=0.001)

def train(epochs):
    train_loss = []
    # minimum = 1e6
    for epoch in range(epochs):
        LSTM_model.train()
        output = LSTM_model(input = train_input,predict_length = 0)
        loss = criterion(output,train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 20 == 0:
            LSTM_model.eval()
            val_predict = LSTM_model(input=val_input[:,:1,:],predict_length = 199)
            val_loss = criterion(val_predict,val_output)
            train_loss.append(loss.tolist())
            print(f"epoch:{epoch},train_loss:{loss},val_loss:{val_loss}")
        torch.save(LSTM_model, "model2")

    return train_loss

# trainLoss = train(401)

# testName = '0.5_0.5_'
testName = '-0.5_0.5_'
testName = '1.0_0.0_'

test_state = torch.Tensor(np.loadtxt(testName+'state').reshape(201,2))
test_input = torch.Tensor(np.loadtxt(testName+'state')[:step,0].reshape(1,1,2))
# print(test_input)
test_result = np.loadtxt(testName+'state')[step:,0].reshape(201-step,1)

predict_result = LSTM_model(input=test_input,predict_length = 201-step)
predict_result = predict_result.detach().numpy()

fig,axes = plt.subplots(2,1)
ax1=axes[0]

ax1.plot(test_result[:,0],label = "real")
ax1.plot(predict_result,label = "predict")
ax1.legend()

plt.show()

print(np.mean(np.square(test_result-predict_result)))


