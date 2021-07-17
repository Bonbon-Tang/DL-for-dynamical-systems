import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

force1 = np.loadtxt('0.0_0.0_input_15%').reshape(300,1,1)
state1 = np.loadtxt('0.0_0.0_state_15%').reshape(301,1,2)
src1 = np.concatenate((state1[:-1,:,:],force1[:,:,:]),axis = 2)
output1 = state1[1:,:,:]

force2 = np.loadtxt('0.0_-1.0_input_15%').reshape(300,1,1)
state2 = np.loadtxt('0.0_-1.0_state_15%').reshape(301,1,2)
src2 = np.concatenate((state2[:-1,:,:],force2[:,:,:]),axis = 2)
output2 = state2[1:,:,:]

force3 = np.loadtxt('1.0_1.0_input_15%').reshape(300,1,1)
state3 = np.loadtxt('1.0_1.0_state_15%').reshape(301,1,2)
src3 = np.concatenate((state3[:-1,:,:],force3[:,:,:]),axis = 2)
output3 = state3[1:,:,:]

force4 = np.loadtxt('2.0_1.5_input_15%').reshape(300,1,1)
state4 = np.loadtxt('2.0_1.5_state_15%').reshape(301,1,2)
src4 = np.concatenate((state4[:-1,:,:],force4[:,:,:]),axis = 2)
output4 = state4[1:,:,:]

train_src = np.concatenate((src1,src2,src3,src4), axis=1)
train_out = np.concatenate((output1,output2,output3,output4), axis=1)
train_src = torch.Tensor(train_src)
train_out = torch.Tensor(train_out)
# print(train_src.shape,train_out.shape)
forceV = np.loadtxt('0.0_1.0_input_15%').reshape(300,1,1)
stateV = np.loadtxt('0.0_1.0_state_15%').reshape(301,1,2)
srcV = np.concatenate((stateV[:100,:,:],forceV[:100,:,:]),axis = 2)
outputV = stateV[100:150,:,:]
val_src = torch.Tensor(srcV)
val_force = torch.Tensor(forceV[100:150,:,:])
val_out = torch.Tensor(outputV)
# print(val_src.shape,val_out.shape,val_force.shape)
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.0, max_len=300):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, intoken, outtoken,final, hidden, nlayers=3, dropout=0.1,steps = 3):
        super(TransformerModel, self).__init__()
        nhead = 4

        self.encoder = nn.Linear(intoken, hidden)
        # self.pos_encoder = PositionalEncoding(hidden, dropout)

        # self.decoder = nn.Linear(outtoken, hidden)
        # self.pos_decoder = PositionalEncoding(hidden, dropout)

        self.encoder_layer = (nn.TransformerEncoderLayer(d_model=hidden,nhead=nhead,
                                                       dim_feedforward=hidden,dropout=dropout))

        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=nlayers)

        self.fc_out = nn.Linear(hidden, final)

        self.src_mask = torch.triu(torch.ones(300, 300), 1)
        self.src_mask = self.src_mask.masked_fill(self.src_mask == 1, float('-inf'))
        for i in range(steps-1,300):
            for j in range(i-steps+1):
                self.src_mask[i,j] = float('-inf')

    # def generate_square_subsequent_mask(self, sz):
    #     mask = torch.triu(torch.ones(sz, sz), 1)
    #     mask = mask.masked_fill(mask == 1, float('-inf'))
    #     return mask

    # def generate_square_subsequent_mask(self, sz):
    #     mask = self.src_mask[:sz,:sz]
    #     print(sz)
    #     print(mask.shape)
    #     return mask

    def forward(self, src):
        src_mask = self.src_mask
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            src_mask = self.src_mask[:len(src),:len(src)]
        src = self.encoder(src)
        output = self.transformer(src,mask = src_mask)
        output = self.fc_out(output)
        return output

def evaluate (length,initial,force):
    encoder_input = initial
    for i in range (length-1):
        # print(i)
        output = Transformer(encoder_input)
        output = output[-1,:,].reshape(1,1,2)
        one_force = force[i,:,:].reshape(1,1,1)
        output = torch.cat((output,one_force),dim = 2)
        encoder_input = torch.cat((encoder_input,output),dim = 0)
    output = Transformer(encoder_input)
    output = output[-1, :, ].reshape(1, 1, 2)
    predict_output = torch.cat((encoder_input[-length+1:,:,:2],output),dim=0)
    return predict_output

Transformer = TransformerModel(intoken=3, outtoken=3, final=2, hidden=20, nlayers=2, dropout=0.05,steps=10)
Transformer = torch.load("model1")

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(Transformer.parameters(),lr=0.001)

# output = evaluate(50,val_src,val_force)
# print(output.shape)

def train(epochs):
    train_loss = []
    # minimum = 1e6
    for epoch in range(epochs):
        Transformer.train()
        output = Transformer(src=train_src)
        loss = criterion(output,train_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 20 == 0:
            Transformer.eval()
            val_predict = evaluate(50,val_src,val_force)
            val_loss = criterion(val_predict,val_out)
            train_loss.append(loss.tolist())
            print(f"epoch:{epoch},train_loss:{loss},val_loss:{val_loss}")
        torch.save(Transformer, "model1")

    return train_loss

# trainLoss = train(801)
torch.save(Transformer, "Transformer_withoutPE_10Pre_15%")

testName = '2.0_1.0_'
testName = '0.0_1.5_'
testName = '0.5_0.5_'


forceT = np.loadtxt(testName+'input_15%').reshape(300,1,1)
stateT = np.loadtxt(testName+'state_15%').reshape(301,1,2)

# srcV = np.concatenate((stateV[:100,:,:],forceV[:100,:,:]),axis = 2)
# outputV = stateV[100:150,:,:]
# val_src = torch.Tensor(srcV)
# val_force = torch.Tensor(forceV[100:150,:,:])
# val_out = torch.Tensor(outputV)

test_src = np.concatenate((stateT[:100,:,:],forceT[:100,:,:]),axis = 2)
test_force = forceT[100:150,:,:]
test_src = torch.tensor(test_src).float()
test_force = torch.tensor(test_force).float()
x_real = np.loadtxt(testName+'state').reshape(301,1,2)[100:150,:,0].reshape(50)
v_real = np.loadtxt(testName+'state').reshape(301,1,2)[100:150,:,1].reshape(50)


predict = evaluate(50,test_src,test_force)
predict = predict.detach().numpy()

x_predict = predict[:,:,0].reshape(50,)
v_predict = predict[:,:,1].reshape(50,)


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
