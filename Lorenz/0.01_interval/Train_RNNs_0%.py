import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

length = 200
xyz1 = np.loadtxt('0.0_0.5_0.0_xyz').reshape(10000//length,length,3)
input1 = xyz1[:,:-1,:]
output1 = xyz1[:,1:,:]
# print(input1.shape,output1.shape)

xyz2 = np.loadtxt('1.0_1.0_1.0_xyz').reshape(10000//length,length,3)
input2 = xyz2[:,:-1,:]
output2 = xyz2[:,1:,:]

xyz3 = np.loadtxt('2.0_2.0_2.0_xyz').reshape(10000//length,length,3)
input3 = xyz3[:,:-1,:]
output3 = xyz3[:,1:,:]

train_input = np.concatenate((input1,input2,input3), axis=0)
train_output = np.concatenate((output1,output2,output3), axis=0)
train_input = torch.Tensor(train_input)
train_output = torch.Tensor(train_output)
# print(train_input.shape,train_output.shape)

xyzV = np.loadtxt('0.0_1.5_1.5_xyz').reshape(10000//length,length,3)
inputV = xyzV[:,:-1,:]
outputV = xyzV[:,1:,:]

val_input = torch.Tensor(inputV)
val_output = torch.Tensor(outputV)

class RNNs(nn.Module):
    def __init__(self, input_size = 3,hidden_size = 32, output_size = 3, n_layers=1):
        super(RNNs, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.input_size = input_size
        self.n_layers = n_layers

        #encoder and decoder have different parameters
        self.rnns = nn.LSTM(input_size, hidden_size, n_layers,batch_first=True)
        # self.decoder = nn.GRU(input_size, hidden_size, n_layers,batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input:torch.Tensor, predict_length = 300):
        hidden = None
        if predict_length == 0:
            encoder_outputs, hidden = self.rnns(input, hidden)

            output = self.out(encoder_outputs)

            return output
        else:
            encoder_outputs, hidden = self.rnns(input, hidden)

            decoder_input = encoder_outputs[:, -1, :].reshape(-1, 1, self.hidden_size)
            decoder_input = self.out(decoder_input)

            decoder_output_set = torch.zeros([0])
            decoder_output_set = torch.cat((decoder_output_set, decoder_input), dim=1)

            for i in range (predict_length-1):
                one_output, hidden = self.rnns(decoder_input, hidden)
                decoder_output = self.out(one_output)
                decoder_output_set = torch.cat((decoder_output_set, decoder_output), dim=1)
                decoder_input = decoder_output
            return decoder_output_set

RNN_model = RNNs()
RNN_model = torch.load("model_LSTM1_0%")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(RNN_model.parameters(),lr=0.01)

def train(epochs):
    train_loss = []
    # minimum = 1e6
    for epoch in range(epochs):
        RNN_model.train()
        output = RNN_model(input = train_input,predict_length = 0)
        loss = criterion(output,train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 20 == 0:
            RNN_model.eval()
            val_predict = RNN_model(input=val_input, predict_length = 0)
            val_loss = criterion(val_predict,val_output)
            train_loss.append(loss.tolist())
            print(f"epoch:{epoch},train_loss:{loss},val_loss:{val_loss}")
        torch.save(RNN_model, "model2")

    return train_loss

# _ = train(1601)

# torch.save(RNN_model, "model_LSTM1_0%")

test_length = 100


# startPoint = 50
# test_input = torch.Tensor(np.loadtxt('0.2_0.2_0.3_xyz')[startPoint,:].reshape(1,1,3))
# test_result = np.loadtxt('0.2_0.2_0.3_xyz')[startPoint+1:startPoint+1+test_length,:]

test_input = torch.Tensor(np.loadtxt('0.2_0.2_0.3_xyz'))
test_result = np.loadtxt('0.2_0.2_0.3_xyz')

def evaluate (test_input,length):
    input = test_input[length * 0, :].reshape(1, 1, 3)
    output = RNN_model(input, length - 1)
    output_set = torch.cat((input,output),dim=1)
    for i in range(1,10000//length):
        input = test_input[length*i,:].reshape(1,1,3)
        output = RNN_model(input,length-1)
        output_set = torch.cat((output_set,input,output),dim=1)
    return output_set

def evaluate2 (test_input,length):
    input = test_input[:length, :].reshape(1,length, 3)
    output = RNN_model(input,length)
    output_set = output
    for i in range(1,10000//length-1):
        input = test_input[i*length:(i+1)*length,:].reshape(1,length, 3)
        output = RNN_model(input,length)
        output_set = torch.cat((output_set,output),dim=1)
    return output_set

predict_result = evaluate(test_input, length = test_length)
predict_result = predict_result.detach().numpy().reshape(-1,3)
# test_result = np.loadtxt('0.2_0.2_0.3_xyz')
test_result = np.loadtxt('0.2_0.2_0.3_xyz')[test_length:,:]

predict_result = evaluate2(test_input, length = test_length)
predict_result = predict_result.detach().numpy().reshape(-1,3)

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



