import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from loadData import readData, loadAllData
from models import GRUNet

predict_mode = 2
is_save = False
LR = 1e-3
epoch = 10
time_step = 30
train_end = 4000
batchsize = 12
# input_column=['date','close','volume']
input_column=['date','close']
dims = len(input_column)-1

pred_col=['close']
path = './data/stock_data.csv'
# 数据集建立
series_train, series_test, raw_data, std, mean = readData(path, input_column,pred_col, time_step=time_step, train_split=train_end)
# df, df_all, df_index

trainset = loadAllData(series_train)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
testset = loadAllData(series_test)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

rnn = GRUNet(dims)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

for step in range(epoch):
    for data, label in trainloader:
        data, label = torch.tensor(data).float(), torch.tensor(label).float()
        pred = rnn(data)
        loss = loss_func(torch.squeeze(pred), label)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()
    print(step, loss)
    if step % 10 and is_save:
        torch.save(rnn, 'rnn.pkl')

generate_data_train = []
generate_data_test = []
predict_times = None

test_index = len(raw_data) + train_end
print("predict mode:",predict_mode)
if predict_mode==1:
    # 预测方法 1：用真实的未来值，逐日预测（相当于每日都有真实值更新data
    predict_times = len(testset)
    eval_pred = []
    for data, label in testloader:
            data, label = torch.tensor(data).float(), torch.tensor(label).float()
            pred = rnn(data)
            eval_pred.append(torch.squeeze(pred).detach().numpy()*std + mean)
else:
    # 预测方法 2：没有真实未来值，逐日预测，并将其更新到data，进行下一天的预测
    predict_times = 1000
    for data, label in testloader:
        break
    data = torch.tensor(data).float()
    eval_pred = []
    assert dims == 1, "features dim should == 1!"
    for i in range(predict_times):
        pred = rnn(data)
        eval_pred.append(torch.squeeze(pred).detach().numpy()*std + mean)
        pred_append = pred.detach().unsqueeze(dim=-1)
        data = torch.cat([data[:,1:,:], pred_append], dim=1)

tag = 'close'
date_index = raw_data.index.tolist()
# plt.plot(date_index[:train_end], generate_data_train, label='real_origin_data(close)')
plt.plot(date_index[train_end: train_end+predict_times], eval_pred, label='predict_{}_value'.format(tag))
plt.plot(date_index[:], raw_data[tag][:], label='real_{}_value'.format(tag))
plt.legend()
plt.show()
