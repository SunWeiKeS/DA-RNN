"""
DA—RNN模型分为两个阶段

第一阶段：encoder
    引入注意力机制，通过参考先前的编码器隐藏状态，
    在每个时间步自适应地提取相关的输入特征。
第二阶段：decoder
    我们使用时间注意力机制在所有时间步中选择相关编码器的隐藏状态

这两个注意力机制都集成在以LSTM为基础的RNN模型，并可以使用标准反向传播一起训练
"""

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from config import device


def init_hidden(x, hidden_size: int):
    """
    Train the initial value of the hidden state:
    batchsize:x.size(0)
    """
    # return Variable(torch.zeros(1, x.size(0), hidden_size))
    return torch.zeros(1, x.shape[0], hidden_size)


class Encoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, T: int):
        """
        input size: number of underlying factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """

        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        # Construct Input Attention Mechanism via deterministic attention model确定性注意力模型
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        # self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1)
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T, out_features=1)

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1)

    def forward(self, input_data):
        # input_data 默认传进来的在gpu上的tensor:   (batch_size,T-1,input_size)
        # input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size))
        # input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size))

        # input_weighted = torch.zeros(input_data.shape[0], self.T - 1, self.input_size).to(device)
        # input_encoded = torch.zeros(input_data.shape[0], self.T - 1, self.hidden_size).to(device)
        input_weighted = torch.zeros(input_data.shape[0], self.T, self.input_size).to(device)
        input_encoded = torch.zeros(input_data.shape[0], self.T, self.hidden_size).to(device)

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)

        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(input_data, self.hidden_size).to(device)  # 1*batch_size*hidden_size
        cell = init_hidden(input_data, self.hidden_size).to(device)

        # for t in range(self.T - 1):  # 对每一个时间步进行操作
        for t in range(self.T):  # 对每一个时间步进行操作
            """
            Eqn. 8: concatenate the hidden states with each predictor 
            batch_size*input_size*(2*hidden_size+T-1)
            文章采用了多层感知机的attn，这里采用的是权重拼接法
            """
            # numpy 转tensor

            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)  # batch_size * input_size * (2*hidden_size + T)
            # Eqn. 8: Get attention weights,x转为二维张量，作为全连接层输入,提取相关输入特征的attention

            # x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1))  # (batch_size * input_size) * 1
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T))  # (batch_size * input_size) * 1

            # Eqn. 9: Softmax the attention weights
            # 这里显然是一个batch做一次softmax(每一个独立时间步),dim=1即每一行一次softmax，为每个inputsize分配权重
            attn_weights = F.softmax(x.view(-1, self.input_size),
                                     dim=1)  # 这里view了一次又一次改变了x的形状，变成了(batch_size, input_size)

            # Eqn. 10: get new input for LSTM
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)

            # 为了提高内存的利用率和效率，调用flatten_parameters让parameter的数据存放成contiguous chunk(连续的块)
            self.lstm_layer.flatten_parameters()

            """
            在0维度增加一个维度 ->  (1, batch_size, input_size)
            hidden shape (1, batch, hidden_size)
            cell of shape (1, batch, hidden_size)
            """
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]

            # save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden

        return input_weighted, input_encoded


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, out_feats=1):
        super(Decoder, self).__init__()
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        # 把encoder里面的attention部分叉开，在中间加了层tanh
        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_hidden_size + encoder_hidden_size, encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(encoder_hidden_size, 1))
        """
        Inputs: input, (h_0, c_0)
                input of shape (seq_len, batch, input_size)
                h_0 of shape (num_layers * num_directions, batch, hidden_size)
                c_0 of shape (num_layers * num_directions, batch, hidden_size)
        Outputs: output, (h_n, c_n)
                output of shape (seq_len, batch, num_directions * hidden_size)
                h_n of shape (num_layers * num_directions, batch, hidden_size)
                c_n of shape (num_layers * num_directions, batch, hidden_size)  
        """
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats)
        self.fc_final = nn.Linear(decoder_hidden_size + encoder_hidden_size, out_feats)

        self.fc.weight.data.normal_()

    def forward(self, input_encoded, y_history):
        # input_encoded:(batch_size,T,encoder_hidden_size)
        # y_history:(batch_size,(T-1))
        # Initialize hidden and cell, (1,batch_size,decoder_hidden_size)
        input_encoded = input_encoded.to(device)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size).to(device)  # 1*input_encoded*decoder_hidden_size
        cell = init_hidden(input_encoded, self.decoder_hidden_size).to(device)

        # context=Variable(torch.zeros(input_encoded.size(0),self.encoder_hidden_size))
        context = torch.zeros(input_encoded.size(0), self.encoder_hidden_size).to(
            device)  # batch_size*encoder_hidden_size

        for t in range(self.T - 1):
            # x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
            #                cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
            #                input_encoded), dim=2)  # (batch_size,T-1,(2*decoder_hidden_size+encoder_hidden_size))
            x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T , 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)  # (batch_size,T,(2*decoder_hidden_size+encoder_hidden_size))
            # Eqn. 12 & 13: softmax on the computed attention weights
            # 在所有时间步选择隐藏状态的attention
            x = self.attn_layer(
                x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size))  # (batch_size*(T))*1

            # 在T-1个时间步，为每一个时间步分配权重
            # beta = F.softmax(x.view(-1, self.T - 1), dim=1)  # (batch_size, T - 1)
            beta = F.softmax(x.view(-1, self.T), dim=1)  # (batch_size, T)

            # Eqn. 14: compute context vector
            # unsqueeze(1) --> (batch_size,1,(T))
            # input_encoded:(batch_size,T,encoder_hidden_size)
            # [:, 0, :] 这里可以理解为降维了，变成batch_size*encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), input_encoded)[:, 0, :]

            # Eqn. 15 --> batch_size*(encoder_hidden_size + out_feats)--> batch_size*out_feats
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # 沿着dim_1，则batch_size不变，hidden_size增加

            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters()
            # 在0维度增加一个维度 ->  (1, batch_size, out_feats)
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1, batch_size, decoder_hidden_size
            cell = lstm_output[1]  # 1, batch_size, decoder_hidden_size

        # Eqn. 22: final output
        """
        hidden[0,:,:] (batch_size, decoder_hidden_size)
        context   (batch_size,encoder_hidden_size)
        cat --> batch_size,decoder_hidden_size + encoder_hidden_size
        """
        return self.fc_final(torch.cat((hidden[0, :, :], context), dim=1))
