import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from loadData import readData, loadAllData
from models import GRUNet


path = './data/stock_data.csv'
# input_column=['date','close','volume']
input_column=['date','close']
pred_col=['close']

# path = './data/601238.csv'
# input_column = ['date','close_front']
# pred_col=['close_front']

predict_mode = 1 # use method 1 or method 2 for predict
predict_times = 100 # only used when predict_mode==2
is_save = False
LR = 1e-3
epoch = 1
time_step = 30
predict_time_step = 1 # 用time_step天 预测未来 predict_time_step 天
train_end = -198
batchsize = 12
dims = len(input_column)-1
# 数据集建立
series_train, series_test, raw_data, std, mean = readData(path, input_column,pred_col, encoding = "gb2312", time_step=time_step,predict_time_step=predict_time_step, train_split=train_end)

trainset = loadAllData(series_train)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
testset = loadAllData(series_test)
testloader = DataLoader(testset, batch_size=1, shuffle=False)
print("train data length:",len(trainset))
print("test data length:",len(testset))
rnn = GRUNet(dims, predict_times_step=predict_time_step)
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

test_index = len(raw_data) + train_end
print("predict mode:",predict_mode)
if predict_mode==1:
    # 预测方法 1：用真实的未来值，逐日预测（相当于每日都有真实值更新data
    predict_times = len(testset)
    eval_pred = []
    for i, (data, label) in enumerate(testloader):
            data, label = torch.tensor(data).float(), torch.tensor(label).float()
            pred = rnn(data)
            eval_pred.append(pred[0][0].unsqueeze(dim=-1).detach().numpy()*std + mean)

else:
    # 预测方法 2：没有真实未来值，逐日预测，并将其更新到data，进行下一天的预测
    for data, label in testloader:
        break
    data = torch.tensor(data).float()
    eval_pred = []
    assert dims == 1, "features dim should == 1!"
    for i in range(predict_times//predict_time_step):
        pred = rnn(data) # [b, predict_time_step]
        eval_pred.append(torch.squeeze(pred).detach().numpy()*std + mean) # 输出数据恢复正则
        pred_append = pred.detach().unsqueeze(dim=-1)
        data = torch.cat([data[:,predict_time_step:,:], pred_append], dim=1)

eval_pred = np.concatenate(eval_pred).squeeze()# 输出数据为[[predict_time_step天],[predict_time_step天]...]需要concat成一列

tag = pred_col[-1]
date_index = raw_data.index.tolist()
# plt.plot(date_index[:train_end], generate_data_train, label='real_origin_data(close)')
plt.plot(date_index[train_end: train_end+predict_times], eval_pred, label='predict_{}_value'.format(tag))
plt.plot(date_index[:], raw_data[tag][:], label='real_{}_value'.format(tag))
plt.legend()
plt.show()
