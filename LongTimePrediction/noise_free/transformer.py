import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import math
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

#src: (observed length, batch, input_features):(1,4,3)
#trg: (1-step before output, batch, input_features):(300,4,3)
#output:(output_length,batch,output_features):(300,4,3)


force1 = np.loadtxt('0.0_0.0_input').reshape(300,1,1)
state1 = np.loadtxt('0.0_0.0_state').reshape(301,1,2)
src1 = np.concatenate((state1[:1,:,:],force1[:1,:,:]),axis = 2)
trg1 = np.concatenate((state1[:-1,:,:],force1[:,:,:]),axis = 2)
output1 = state1[1:,:,:]

force2 = np.loadtxt('0.0_-1.0_input').reshape(300,1,1)
state2 = np.loadtxt('0.0_-1.0_state').reshape(301,1,2)
src2 = np.concatenate((state2[:1,:,:],force2[:1,:,:]),axis = 2)
trg2 = np.concatenate((state2[:-1,:,:],force2[:,:,:]),axis = 2)
output2 = state2[1:,:,:]

force3 = np.loadtxt('1.0_1.0_input').reshape(300,1,1)
state3 = np.loadtxt('1.0_1.0_state').reshape(301,1,2)
src3 = np.concatenate((state3[:1,:,:],force3[:1,:,:]),axis = 2)
trg3 = np.concatenate((state3[:-1,:,:],force3[:,:,:]),axis = 2)
output3 = state3[1:,:,:]

force4 = np.loadtxt('2.0_1.5_input').reshape(300,1,1)
state4 = np.loadtxt('2.0_1.5_state').reshape(301,1,2)
src4 = np.concatenate((state4[:1,:,:],force4[:1,:,:]),axis = 2)
trg4 = np.concatenate((state4[:-1,:,:],force4[:,:,:]),axis = 2)
output4 = state4[1:,:,:]

train_src = np.concatenate((src1,src2,src3,src4), axis=1)
train_trg = np.concatenate((trg1,trg2,trg3,trg4), axis=1)
train_out = np.concatenate((output1,output2,output3,output4), axis=1)
train_src = torch.Tensor(train_src)
train_trg = torch.Tensor(train_trg)
train_out = torch.Tensor(train_out)
# print(train_src.shape,train_trg.shape,train_out.shape)

forceV = np.loadtxt('2.0_1.0_input').reshape(300,1,1)
stateV = np.loadtxt('2.0_1.0_state').reshape(301,1,2)
srcV = np.concatenate((stateV[:1,:,:],forceV[:1,:,:]),axis = 2)
trgV = np.concatenate((stateV[:-1,:,:],forceV[:,:,:]),axis = 2)
outputV = stateV[1:,:,:]
val_src = torch.Tensor(srcV)
val_trg = torch.Tensor(trgV)
val_out = torch.Tensor(outputV)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=300):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, intoken, outtoken,final, hidden, nlayers=3, dropout=0.1):
        super(TransformerModel, self).__init__()
        nhead = 4

        self.encoder = nn.Linear(intoken, hidden)
        self.pos_encoder = PositionalEncoding(hidden, dropout)

        self.decoder = nn.Linear(outtoken, hidden)
        self.pos_decoder = PositionalEncoding(hidden, dropout)

        self.inscale = math.sqrt(intoken)
        self.outscale = math.sqrt(outtoken)

        self.transformer = nn.Transformer(d_model=hidden, nhead=nhead, num_encoder_layers=nlayers,
                                          num_decoder_layers=nlayers, dim_feedforward=hidden, dropout=dropout)
        self.fc_out = nn.Linear(hidden, final)

        self.src_mask = None
        self.trg_mask = None
        self.memory_mask = None

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def make_len_mask(self, inp):
        return (inp == 0).transpose(0, 1)

    def forward(self, src, trg):
        #input: time_length,batch,features
        if self.trg_mask is None or self.trg_mask.size(0) != len(trg):
            self.trg_mask = self.generate_square_subsequent_mask(len(trg))

        src_pad_mask = self.make_len_mask(src)
        trg_pad_mask = self.make_len_mask(trg)

        src = self.encoder(src)
        src = self.pos_encoder(src)

        trg = self.decoder(trg)
        trg = self.pos_decoder(trg)
        output = self.transformer(src, trg, tgt_mask=self.trg_mask)
        # output = self.transformer(src, trg, src_mask=self.src_mask, tgt_mask=self.trg_mask,
        #                           memory_mask=self.memory_mask,
        #                           src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=trg_pad_mask,
        #                           memory_key_padding_mask=src_pad_mask)
        output = self.fc_out(output)

        return output

Transformer  = TransformerModel(intoken=3, outtoken=3,final=2, hidden=20, nlayers=1, dropout=0.0)
Transformer = torch.load("model1")
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(Transformer.parameters(),lr=0.005)

def train(epochs):
    train_loss = []
    # minimum = 1e6
    for epoch in range(epochs):
        Transformer.train()
        output = Transformer(src=train_src,trg=train_trg)
        loss = criterion(output,train_out)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 20 == 0:
            Transformer.eval()
            val_predict = Transformer(src=val_src,trg=val_trg)
            val_loss = criterion(val_predict,val_out)
            train_loss.append(loss.tolist())
            print(f"epoch:{epoch},train_loss:{loss},val_loss:{val_loss}")
        torch.save(Transformer, "model1")

    return train_loss

trainLoss = train(600)
# torch.save(Transformer, "Transformer_withoutPE")


def evaluate (length):
    decoder_input = val_src
    for i in range (length):
        output = Transformer(val_src,decoder_input)
        output = output[-1,:,].reshape(1,1,2)
        force = val_trg[i,:,-1].reshape(1,1,1)
        output = torch.cat((output,force),dim = 2)
        decoder_input = torch.cat((decoder_input,output),dim = 0)
    return decoder_input[1:,:,:]

val_predict = evaluate(300)

# val_predict = Transformer(src=val_src, trg=val_trg)
val_predict = val_predict.detach().numpy()
x_predict = val_predict[:,:,0].reshape(300,)
v_predict = val_predict[:,:,1].reshape(300,)
x_real = val_out[:,:,0].reshape(300,).numpy()
v_real = val_out[:,:,1].reshape(300,).numpy()

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
