import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

startPoint = 5

state1 = np.loadtxt('1.0_2.0_state')[:,0].reshape(1,201,1)
encoder_input1 = state1[:,:startPoint,:]
decoder_input1 = state1[:,startPoint:-1,:]
output1 = state1[:,startPoint:,:]


state2 = np.loadtxt('2.0_0.5_state')[:,0].reshape(1,201,1)
encoder_input2 = state2[:,:startPoint,:]
decoder_input2 = state2[:,startPoint:-1,:]
output2 = state2[:,startPoint:,:]

state3 = np.loadtxt('-2.0_1.0_state')[:,0].reshape(1,201,1)
encoder_input3 = state3[:,:startPoint,:]
decoder_input3 = state3[:,startPoint:-1,:]
output3 = state3[:,startPoint:]

state4 = np.loadtxt('-1.0_-1.0_state')[:,0].reshape(1,201,1)
encoder_input4 = state4[:,:startPoint,:]
decoder_input4 = state4[:,startPoint:-1,:]
output4 = state4[:,startPoint:,:]

train_en_input = np.concatenate((encoder_input1,encoder_input2,encoder_input3,encoder_input4), axis=0)
train_de_input = np.concatenate((decoder_input1,decoder_input2,decoder_input3,decoder_input4), axis=0)
train_output = np.concatenate((output1,output2,output3,output4), axis=0)
train_en_input = torch.Tensor(train_en_input)
train_de_input = torch.Tensor(train_de_input)
train_output = torch.Tensor(train_output)
# print(train_en_input.shape,train_de_input.shape,train_output.shape)

stateV = np.loadtxt('0.0_1.0_state')[:,0].reshape(1,201,1)
encoder_inputV = stateV[:,:startPoint,:]
decoder_inputV = stateV[:,startPoint:-1,:]
outputV = stateV[:,startPoint:]
val_en_input = torch.Tensor(encoder_inputV)
val_de_input = torch.Tensor(decoder_inputV)
val_output = torch.Tensor(outputV)


class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=48, output_size=1, n_layers=1):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.n_layers = n_layers

        # encoder and decoder have different parameters
        self.encoder = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)

        # self.gru = nn.GRU(input_size, hidden_size, n_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, en_input: torch.Tensor, de_input = None, predict_length=0):
        hidden = None
        if predict_length == 0:
            encoder_outputs, hidden = self.encoder(en_input, hidden)

            encoder_last = self.out(encoder_outputs)[:, -1, :].reshape(-1, 1, 1)

            decoder_outputs,hidden = self.decoder(de_input, hidden)

            decoder_outputs = self.out(decoder_outputs)

            output = torch.cat((encoder_last, decoder_outputs), dim=1)

            return output
        else:
            total_sequence = en_input

            encoder_outputs, hidden = self.encoder(en_input, hidden)

            decoder_input = encoder_outputs[:, -1, :].reshape(-1, 1, self.hidden_size)

            decoder_input = self.out(decoder_input)

            total_sequence = torch.cat((total_sequence, decoder_input),dim=1)

            one_input = decoder_input
            for i in range(predict_length - 1):
                # one_input = total_sequence[-1].reshape(-1, 1, 1)
                one_output, hidden = self.decoder(one_input, hidden)
                one_output = self.out(one_output)
                total_sequence = torch.cat((total_sequence, one_output),dim=1)
                one_input = one_output

            return total_sequence[:,-predict_length:,:]

LSTM_model = RNN()
# LSTM_model = torch.load("model1")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(LSTM_model.parameters(),lr=0.01)

def train(epochs):
    train_loss = []
    # minimum = 1e6
    for epoch in range(epochs):
        LSTM_model.train()
        output = LSTM_model(en_input = train_en_input,de_input = train_de_input,predict_length = 0)
        loss = criterion(output,train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 20 == 0:
            LSTM_model.eval()
            val_predict = LSTM_model(en_input=val_en_input,de_input = val_de_input,predict_length = 0)
            val_loss = criterion(val_predict,val_output)
            train_loss.append(loss.tolist())
            print(f"epoch:{epoch},train_loss:{loss},val_loss:{val_loss}")
        torch.save(LSTM_model, "model2")

    return train_loss

trainLoss = train(801)

testName = '0.5_0.5_'
# testName = '-0.5_0.5_'
# testName = '1.0_0.0_'

test_state = torch.Tensor(np.loadtxt(testName+'state').reshape(201,2))
test_input = torch.Tensor(np.loadtxt(testName+'state')[:startPoint,0].reshape(1,startPoint,1))
test_result = np.loadtxt(testName+'state')[startPoint:,0].reshape(201-startPoint,1)

predict_result = LSTM_model(en_input=test_input,predict_length = 201-startPoint)
predict_result = predict_result.detach().numpy()

fig,axes = plt.subplots(2,1)
ax1=axes[0]

ax1.plot(test_result[:,0],label = "real")
ax1.plot(predict_result[:,:,:].reshape(-1),label = "predict")
ax1.legend()

plt.show()

print(np.mean(np.square(test_result-predict_result)))


