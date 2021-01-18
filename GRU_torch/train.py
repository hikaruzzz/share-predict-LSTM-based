import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

from loadData import readData, loadAllData
from models import GRUNet


# path = './data/stock_data.csv'
# input_column=['date','close','volume']
# # input_column=['date','close']
# pred_col=['close']

path = './data/601238.csv'
input_column = ['date','close','volume']
pred_col=['close']

predict_mode = 2 # use method 1 or method 2 for predict
is_save = False
LR = 1e-3
epoch = 30
time_step = 30
predict_time_step = 30 # 用time_step天 预测未来 predict_time_step 天
predict_times = 60 # only used when predict_mode==2, total predict times, should be N*predict_time_step

# train_end = 400 # train_end==-1 means predict last "predict_time_step" days
train_end = -31 # train_end==-1 means predict last "predict_time_step" days, if use "-", -31
is_train = True

batchsize = 12
dims = len(input_column)-1
# 数据集建立
series_train, series_test, raw_data, std, mean = readData(path, input_column,pred_col, encoding = "gb2312", time_step=time_step,predict_time_step=predict_time_step, train_split=train_end)
torch.manual_seed(888)
trainset = loadAllData(series_train)
trainloader = DataLoader(trainset, batch_size=batchsize, shuffle=True)
testset = loadAllData(series_test)
testloader = DataLoader(testset, batch_size=1, shuffle=False)
print("train data length:",len(trainset))
print("test data length:",len(testset))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

rnn = GRUNet(dims, out_channel=dims, predict_times_step=predict_time_step).to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()
date_index = raw_data.index.tolist()
print("total data:",len(date_index))
assert date_index[1] < date_index[2], "data error: date should be in ascending order"

if is_train:
    for step in range(epoch):
        for data, label in trainloader:
            # data, label = torch.tensor(data).to(device).float(), torch.tensor(label).to(device).float()
            data, label = data.to(device).float(), label.to(device).float()

            pred = rnn(data)
            loss = loss_func(pred, label)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # back propagation, compute gradients
            optimizer.step()
        print(step,"loss", loss.detach())
        if step % 10 and is_save:
            torch.save(rnn, 'rnn.pkl')

generate_data_train = []
generate_data_test = []

predict_idx = input_column.index(pred_col[0])-1
test_index = len(raw_data) + train_end
print("predict mode:",predict_mode)
eval_pred = []
if predict_mode==1:
    # 预测方法 1：用真实的未来值，逐日预测（相当于每日都有真实值更新data
    predict_times = len(testset)
    for i, (data, label) in enumerate(testloader):
            data, label = data.to(device).float(), label.to(device).float()
            pred = rnn(data).to(device)
            eval_app = pred[0][0].detach().numpy()*std + mean
            eval_pred.append(eval_app[predict_idx])
else:
    # 预测方法 2：没有真实未来值，逐日预测，并将其更新到data，进行下一天的预测
    assert len(testloader) != 0, "error: nums of testloader should > 0"
    for data, label in testloader:
        data = data.to(device).float()
        break
    for i in range(predict_times//predict_time_step):
        pred = rnn(data) # [b, predict_time_step]
        eval_pred.append(torch.squeeze(pred,dim=0)[:,predict_idx].detach().cpu().numpy()*std + mean) # 输出数据恢复正则
        pred_append = pred.detach()
        data = torch.cat([data[:,predict_time_step:,:], pred_append], dim=1)
    eval_pred = np.concatenate(eval_pred).squeeze() # 输出数据为[[predict_time_step天],[predict_time_step天]...]需要concat成一列
tag = pred_col[-1]
print(date_index[train_end])
print(len(eval_pred))
# plt.plot(date_index[:train_end], generate_data_train, label='real_origin_data(close)')
assert train_end<0 and train_end-predict_time_step+len(eval_pred)<0, "len(eval_pred) over large, abs(train_end-predict_time_step) should > preidct_times"
plt.title("Loss:{:.3f},epoch:{}".format(loss.detach(),epoch))
plt.plot(date_index[train_end-predict_time_step: train_end-predict_time_step+len(eval_pred)], eval_pred, label='predict_{}_value'.format(tag))
plt.plot(date_index[:], raw_data[tag][:], label='real_{}_value'.format(tag))
plt.legend()
plt.show()
