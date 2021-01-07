import pandas as pd
import numpy as np
import datetime
import torch

def generate_series_n_days(series, time_step, predict_time_step, pred_idx):
    '''
    used by dataloader, output pd list
    :param series: a numpy list of all data on 1 feature
    :param time_step: time_step
    :return: ret_y = [length, ]

    '''
    # time_step coloums and 1 coloums(y)
    ret_x = []
    ret_y = []
    for i in range(time_step):
        if (predict_time_step == 1):
            ret_x.append(series[i:-(time_step - i)])
        else:
            ret_x.append(series[i:-(time_step + predict_time_step - i)])

    for i in range(predict_time_step):
        if(predict_time_step==1):
            ret_y.append(series[time_step:])
        else:
            ret_y.append(series[time_step + i : -(predict_time_step-i)])
    # print(np.array(ret_x).shape, np.array(ret_y).shape)
    return np.array(ret_x).transpose([1, 0, 2]), np.array(ret_y).transpose([1,0,2])


def readData(path, column=['date','close','volume'],pred_col=['close'],encoding='utf-8', time_step=30, predict_time_step=1, train_split=500):
    # df.index is date
    # pred_col should be in column
    # predict 1 dim
    raw_data = pd.read_csv(path, usecols=column,encoding = encoding)
    data, column_name = raw_data[column[1:]].values, raw_data.columns.tolist()
    pred_idx = column.index(pred_col[0])-1 # predict 1 dim

    mean = np.mean(data, axis=0)  # 数据的均值和方差
    std = np.std(data, axis=0)
    norm_data = (data - mean) / std  # 归一化，去量纲

    raw_data.index = list(map(lambda x: datetime.datetime.strptime(x, "%Y/%m/%d"), raw_data['date']))
    data_train = norm_data[:train_split]
    data_test = norm_data[train_split - (time_step + predict_time_step):] # 这个要与input ret_x.append 一样大小
    series_train = generate_series_n_days(data_train, time_step, predict_time_step, pred_idx)
    series_test = generate_series_n_days(data_test, time_step, predict_time_step, pred_idx)

    return series_train, series_test, raw_data, std[pred_idx], mean[pred_idx]

class loadAllData(torch.utils.data.Dataset):
    def __init__(self, data):
        # 定义好 image 的路径
        self.data, self.label = data[0], data[1]

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    root = './data/stock_data.csv'
    train_data, test_data, idx = readData(root)
    # train_data.shape = (total num, time_step+1)
    train_x, train_y = train_data
    print(train_x.shape)


    # tensor = TensorDataset(train_x, train_y)
    # print(tensor)
