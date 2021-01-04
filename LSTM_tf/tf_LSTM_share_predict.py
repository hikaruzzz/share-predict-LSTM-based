import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# paras save dir
paras_dir = './LSTM_data/model_4_share_1'
# error: 'Restoring from checkpoint failed. This is most likely due to a Variable name or other graph key that is missing from the checkpoint'
# resolve: delete paras_dir file

# const paras
time_step = 50  # 用n步长来预测下一步 【序列模型，不需要传入时刻key，只需要靠值val预测下一刻的值】
input_size = 1  # 输入变量维度（可以多维变量）
output_size = 1
learn_rate = 0.0006
h_unit = 200  # 隐层 神经元数量 【理论上应该受time_step的限制？必须大于time_step?】
max_iter = 4000
train_num = 758  # 这个值相当于从第n个数据集开始预测【前n个用于train】 【注意，train后若往大调此值，预测准确率大幅下降（未知数据输入）
pred_day = 30   # 调参注意：【大于 time_step 后的区间可信度低】

#layer_num = 2 # 深层RNN层数
#batch = 55  # 值为 train_num - time_step - 1
'''
lstm_input.shape = [?,time_step,input_dim]
? = batch_size * time_step /time_step / input_dim = lstm_input.shape
例如：train_x
 [[[1],[2],[3],[4]]
 [[2],[3],[4],[5]]]
'''

# set of feed_dict X,Y
X = tf.placeholder(tf.float32,[None,time_step,input_size])
y = tf.placeholder(tf.float32,[None,output_size])

# weight & baise
w_in = tf.Variable(tf.random_normal([input_size,h_unit]))
w_out = tf.Variable(tf.random_normal([h_unit,output_size]))

b_in = tf.Variable(tf.constant(0.1,shape=[h_unit,]))
b_out = tf.Variable(tf.constant(0.1,shape=[output_size,]))  # baise不用梯度下降


def read_share_csv(is_show=0):
    '''
    :param is_show:
    :return: data shape=(?,1)
    '''
    path = './share_data/'
    #file_name = '999999_2.txt'
    file_name = '000001(140318-190516).txt'
    data = pd.read_csv(path+file_name,sep='\t',encoding='GB2312').iloc[:,:].values

    x = np.array([x for x in range(len(data[:,0]))]).reshape([-1,1])
    #data = np.append(x,data,axis=-1)

    if is_show:
        plt.figure(figsize=(18,8))
        plt.plot(data,label='code = 000001')
        plt.legend()
        plt.show()

    return data


def data_generator():
    # return train_x,train_y,test_x,test_y
    '''
    复合函数 数据
    '''
    # train_num = 120  # 前60个为train，剩下为test  【实际train_x = train_num - time_step -1】
    # max_x = 2000
    # x = np.array(range(max_x))
    # y_val = np.sin(np.pi*x/50)+np.cos(np.pi*x/50)+np.sin(np.pi*x/25)+np.random.uniform(-0.2,0.2,max_x)
    # y在（-1，1）区间所以不用标准化

    '''
    share 数据 
    '''

    _y_val = read_share_csv()[500:]  # 取数据
    min_max_scaler = MinMaxScaler([0,1])
    y_val = min_max_scaler.fit_transform(_y_val)

    '''test data
    '''
    #train_num = 10
    #y_val = [x for x in range(20)]

    train_x,train_y,test_x,test_y = [],[],[],[]
    # create train set
    for i in range(train_num - time_step):
        train_x.append(y_val[i:i+time_step])  # 取np.sin(x) 前time_step个作为输入
        train_y.append(y_val[i+time_step])  # 取np.sin(x) 第time_step 作为要预测值

    # create test set
    # 注意，test_x的第一行，所有元素应该都在train_x存在，若第一行最后一个是train_Y中而不是train_x，相当于传入一个未知的数，导致预测第一步就bug
    #解决办法，test_x的第一行 == train_x的最后一行
    j = 0
    for j in range(train_num-time_step-1,len(y_val) - time_step):
        test_x.append(y_val[j:j+time_step])
        test_y.append(y_val[j+time_step])   # 最后一个test_y 是y_val的最后一个
    test_x.append(y_val[j:j+time_step])  # 最后还有一行test_x，但没有test_y

    return train_x,train_y,test_x,test_y,_y_val,min_max_scaler


# struct of lstm
# input layer
def LSTM():
    # multi layer
    cell = tf.nn.rnn_cell.LSTMCell(h_unit,forget_bias=0.8) # 用 BasicRNNCells ，则output.shape = （？，30)
    #init_state = cell.zero_state(batch_size, dtype=tf.float32)
    #_output, _ = tf.nn.dynamic_rnn(cell, X, initial_state=init_state, dtype=tf.float32)

    _output, _ = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)
    output = _output[:, -1, :]  # 只保留第二维的最后一个行[最后一位神经元输出（预测位）] ，(?,4,100)->(?,100)

    # output layer
    # 1 . 用全连接层提取特征  【注意：切换这层结构，需要把check_point文件删除，否则读取参数不一致bug】
    pred = tf.contrib.layers.fully_connected(output,output_size,activation_fn=None)
    # 2. 用简单output结构
    #pred = tf.matmul(output,w_out) + b_out

    return pred


def train(train_x,train_y):
    pred = LSTM()
    loss = tf.losses.mean_squared_error(labels=y,predictions=pred)
    train_opt = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss)
    # opt = tf.contrib.layers.optimize_loss(loss,)

    saver = tf.train.Saver()  # 保存，恢复，checkpoint参数
    with tf.Session() as sess:
        check_point = tf.train.latest_checkpoint(paras_dir)

        if check_point:
            saver.restore(sess,check_point)  # 若有存档，则从paras_dir中恢复参数
        else:
            sess.run(tf.global_variables_initializer())  # 无存档，则初始化参数

        count_temp = 0
        while True:
            count_temp+=1

            train_x = np.reshape(train_x, [-1,time_step, input_size])
            train_y = np.reshape(train_y,[-1,1])

            _, _loss = sess.run([train_opt,loss],feed_dict={X:train_x,y:train_y})

            if count_temp%100==0:
                print(count_temp," = ",_loss)

            if count_temp%1000==0:
                saver.save(sess,save_path=paras_dir+'/model_sinx',global_step=count_temp)

            if count_temp >= max_iter:
                print("train finish")
                saver.save(sess,save_path=paras_dir+'/model_sinx',global_step=count_temp)  # 保存第max_iter步的参数
                break
        # plt.figure(figsize=(10,8))
        # plt.plot(train_y,label='train_y')
        # pred_val = sess.run(pred,feed_dict={X:train_x})
        # plt.plot(pred_val,label='pre_v')
        # plt.show()


def predict_true(train_x,train_y,test_x,data_raw,min_max_scaler):
    # 真实预测，按照不断predict的值来预测下一步，而不是按test_x值
    pred_y = LSTM()  # LSTM 的predict只需要feed x值
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # restore paras
        check_point = tf.train.latest_checkpoint(paras_dir)
        saver.restore(sess, check_point)
        # predict
        test_x = np.reshape(train_x, [-1, time_step, input_size])
        input_x = test_x[-1, :, :]

        pred_val = []
        for i in range(pred_day):
            pre = sess.run(pred_y, feed_dict={X: [input_x]})  # 用test_x第一行，predict 出一个值
            input_x = np.append(input_x[1:,:],pre,axis=0)
            pred_val.append(pre[-1])

        pred_val = min_max_scaler.inverse_transform(pred_val)

        print("data size:",data_raw.shape[0])
        # show
        plt.figure(figsize=(15, 8))
        #plt.plot([None for _ in range(len(data_raw))] + [x for x in pred_val], label='predict_day_'+str(pred_day))
        plt.plot([None for _ in range(len(train_x)+time_step-1)] + [x for x in pred_val],label='pre_val')
        #plt.xticks(np.linspace(0, 1000, 10))
        plt.plot(data_raw,label='train_x')
        plt.legend()
        plt.show()


def predict(train_x,train_y,test_x,data_raw,min_max_scaler):
    pred_y = LSTM()  # LSTM 的predict只需要feed x值
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # restore paras
        check_point = tf.train.latest_checkpoint(paras_dir)
        saver.restore(sess, check_point)
        # predict

        test_x = np.reshape(test_x,[-1,time_step,input_size])
        pred_val = sess.run(pred_y, feed_dict={X: test_x})  #[:4,:,:]
        pred_val = min_max_scaler.inverse_transform(pred_val)
        train_y = min_max_scaler.inverse_transform(train_y)

        # show
        plt.figure(figsize=(15, 8))
        #plt.plot([None for _ in range(time_step)] + [x for x in train_y],label='train_Y')
        plt.plot([None for _ in range(len(train_x)+time_step-2)] + [x for x in pred_val],label='predict')
        plt.plot(data_raw,label='reality data')
        plt.legend()
        plt.show()


train_x,train_y,test_x,test_y,data_raw,min_max_scaler = data_generator()

def Predictor():
    predict_true(train_x, train_y, test_x, data_raw, min_max_scaler)

def Trainer():
    train(train_x,train_y)

if __name__ == '__main__':
    # 注释掉 train 再 predict【tensorflow共享内存导致rnn kernel冲突问题】
    # 通过 paras_dir 文件保存，恢复check point 来避免冲突

    train(train_x, train_y)
    #predict_true(train_x, train_y, test_x, data_raw, min_max_scaler)
    try:
        # tf.reset_default_graph() # 未知具体用法
        predict(train_x, train_y, test_x, data_raw, min_max_scaler)
    except:
        pass