## 2019年5月17日更新

# 基于LSTM 的上证指数预测

# 模型结构
  单层LSTM，隐层神经元数量：200，LSTM的输出结果output.shape=（？，200，1），将之传入一个全连接层（单层）并输出一个值（shape=(?,1))。【?可以看成是一个时间序列长度】
  loss函数：mean_squared_error

# 实验
## 模型拟合度
  此算法的predict函数，采用断点开始预测的方法，从预测点开始不再输入test数据集，而是用预测的值输入进行下一步预测。
  fig.1是拟合度测试结果，从起点开始不再用test测试集，可以看出predict曲线与真实数据集的曲线较拟合，由此可得模型对数据的特征学习效果较拟合。loss = 0.0002  
  ![fig.1](https://github.com/hikaruzzz/share-predict-LSTM-based/blob/master/fig1.png)
## 模型预测效果
  采用80天步长（time_step）预测，训练数据集从2014年至今，由fig.2可以看出，过拟合现象较明显，呈现出重复的数据特征预测。  
  ![fig.2](https://github.com/hikaruzzz/share-predict-LSTM-based/blob/master/fig2.png)
## 预测结果时效性
  根据调参与实验经验，使用50天的步长进行train与predict，30天内的预测数据存在一定准确度。
  如fig.3所示，训练集设定在红箭头之前，对于模型来说红箭头之后的数据为未知数据，并从该时刻开始预测200天（将最近10天数据作为前段部分预测的输入值），可以看出在30天内（红色框内）的预测与真实数据值接近，存在一定准确度。  
  ![fig.3](https://github.com/hikaruzzz/share-predict-LSTM-based/blob/master/fig3.png) 
## 未来预见
  如fig.4所示，预测开始时间为2019年5月17日，预测200天内的走势，等待验证。  
  ![fig.4](https://github.com/hikaruzzz/share-predict-LSTM-based/blob/master/fig4.png)
## 改进方向
  设定一个阈值，当预测的值与当前真实值的差超过该阈值，则停止预测并传入新数据进行重新训练，防止出现fig.5的情况（红框内），由于突然的大幅度变化导致预测结果准确度剧减。  
  ![fig.5](https://github.com/hikaruzzz/share-predict-LSTM-based/blob/master/fig5.png)
## 结论
  时间序列类型数据的预测，对于股票这种波动幅度巨大的数据，预测效果不理想。
        
        
