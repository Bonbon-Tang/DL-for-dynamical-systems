import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

length = 200
xyz1 = np.loadtxt('0.0_0.5_0.0_xyz').reshape(10000//length,length,3)
input1 = xyz1[:,:-1,:]
output1 = xyz1[:,1:,:]
input1 = input1.transpose(1,0,2)
output1 = output1.transpose(1,0,2)

# print(input1.shape,output1.shape)

xyz2 = np.loadtxt('1.0_1.0_1.0_xyz').reshape(10000//length,length,3)
input2 = xyz2[:,:-1,:]
output2 = xyz2[:,1:,:]
input2 = input2.transpose(1,0,2)
output2 = output2.transpose(1,0,2)

xyz3 = np.loadtxt('2.0_2.0_2.0_xyz').reshape(10000//length,length,3)
input3 = xyz3[:,:-1,:]
output3 = xyz3[:,1:,:]
input3 = input3.transpose(1,0,2)
output3 = output3.transpose(1,0,2)

train_input = np.concatenate((input1,input2,input3), axis=1)
train_output = np.concatenate((output1,output2,output3), axis=1)
train_input = torch.Tensor(train_input)
train_output = torch.Tensor(train_output)

xyzV = np.loadtxt('0.0_1.5_1.5_xyz').reshape(10000//length,length,3)
inputV = xyzV[:,:-1,:]
outputV = xyzV[:,1:,:]
inputV = inputV.transpose(1,0,2)
outputV = outputV.transpose(1,0,2)

val_input = torch.Tensor(inputV)
val_output = torch.Tensor(outputV)

class TransformerModel(nn.Module):
    def __init__(self, intoken, finalToken, hidden, nlayers=3, dropout=0.0,steps = 2):
        super(TransformerModel, self).__init__()
        nhead = 4

        self.encoder = nn.Linear(intoken, hidden)

        self.encoder_layer = (nn.TransformerEncoderLayer(d_model=hidden,nhead=nhead,
                                                       dim_feedforward=hidden,dropout=dropout))

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.fc_out = nn.Linear(hidden, finalToken)

        self.src_mask = torch.triu(torch.ones(500, 500), 1)
        self.src_mask = self.src_mask.masked_fill(self.src_mask == 1, float('-inf'))
        for i in range(steps - 1, 500):
            for j in range(i - steps + 1):
                self.src_mask[i, j] = float('-inf')

    # def generate_square_subsequent_mask(self, sz):
    #     mask = torch.triu(torch.ones(sz, sz), 1)
    #     mask = mask.masked_fill(mask == 1, float('-inf'))
    #     return mask
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        for i in range(4,sz):
            for j in range(i-4):
                mask[i,j] = float('-inf')
        return mask

    def forward(self, src):
        src_mask = self.src_mask
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            src_mask = self.src_mask[:len(src), :len(src)]
        src = self.encoder(src)
        output = self.transformer(src, mask=src_mask)
        output = self.fc_out(output)
        return output

def Predict (length,initial):
    encoder_input = initial
    for i in range (length):
        output = model_transformer(encoder_input)
        output = output[-1,:,].reshape(1,1,3)
        encoder_input = torch.cat((encoder_input,output),dim = 0)
    return encoder_input[-length:,:,:]

def wholePrediction(length,test_input):
    input = test_input[length * 0, :].reshape(1, 1, 3)
    output = Predict(length - 1,input)
    output_set = torch.cat((input, output), dim=0)
    for i in range(1, 10000 // length):
        input = test_input[length * i, :].reshape(1, 1, 3)
        output = Predict(length - 1,input)
        output_set = torch.cat((output_set, input, output), dim=0)
    return output_set

model_transformer = TransformerModel(intoken=3,finalToken=3,hidden=32,nlayers=1,steps=5,dropout=0.0)
# model_transformer = torch.load("model2")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_transformer.parameters(),lr=0.1)

# print(train_input.shape)
# print(model_transformer(train_input).shape)
# output = Predict(100,train_input[:1,:,:])
# print(output.shape)

def train(epochs):
    train_loss = []
    # minimum = 1e6
    for epoch in range(epochs):
        model_transformer.train()
        output = model_transformer(train_input)
        loss = criterion(output,train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 20 == 0:
            model_transformer.eval()
            val_predict = model_transformer(val_input)
            val_loss = criterion(val_predict,val_output)
            train_loss.append(loss.tolist())
            print(f"epoch:{epoch},train_loss:{loss},val_loss:{val_loss}")
        torch.save(model_transformer, "model1")

    return train_loss

_ = train(801)
# torch.save(model_transformer, "model_Transformer5_0%")

test_input = torch.Tensor(np.loadtxt('0.2_0.2_0.3_xyz'))
test_result = np.loadtxt('0.2_0.2_0.3_xyz')

PredictLength = 20

predict_result = wholePrediction(PredictLength,test_input)
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