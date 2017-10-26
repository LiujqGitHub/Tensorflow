import tensorflow as tf
import numpy as np
# https://zhuanlan.zhihu.com/p/28196873

### BasicRNNCell使用
# description: 学习单步的RNN:RNNCell，它是RNN的基本单元，每个RNNCell都有个call方法
# 使用方式是：(output, next_state) = call(input, state);假设我们有个初始状态h0,还有
# 输入x1,调用call(x1, h0)后就可以得到(output1, h1);再调用一次call(x2,h1)就可以得到
# (output2, h2);话句话说，每一次调用RNNCell的call方法，就相当于在时间上推进了一步，这就是
# RNNCell的基本功能。在代码实现上，RNNCell只是一个抽象类，我们用的时候都是用的它的两个子类
# BasicRNNCell、BasicLSTMCell。故名思义，前者是RNN的基础类，后者是LSTM的基础类。

# 除了call方法外，对于RNNCell，还有两个类属性比较重要：state_size(隐层大小)
# output_size(输出大小)。我们通常是将一个batch送到模型计算，设输入数据的形状为
# (batch_size, input_size)，那么计算时得到的隐层状态就是(batch_size,state_size)
# 输出的就是(batch_size, output_size)

# cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size=128
# print (cell.state_size) # 128
# inputs = tf.placeholder(np.float32, shape=(32,100)) # 32 是batch_size
# h0 = cell.zero_state(32,np.float32) # 通过zero_state得到一个全0的初始状态，形状为（batch_size, state_size）
# output, h1 = cell.call(inputs,h0)
# print iim(h1.shape) # (32, 128)


###BasicLSTMCell使用

# LSTM可以看做有两个隐状态h和c，对应的隐层就是一个Tuple,每个都是（batch_size,state_size）的形状
# lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
# inputs = tf.placeholder(np.float32, shape=(32,100)) # 32 是 batch_size
# h0 = lstm_cell.zero_state(32, np.float32)
# output, h1 = lstm_cell.call(inputs, h0)
# print (h1.h)
# print (h1.c)



### 一次执行多步：tf.nn.dynamic_rnn
# 如果我们的序列长度为10， 就要调用10次call函数，比较麻烦，对此，Tensorflow提供了一个tf.nn.dynamic_rnn函数
# 使用该函数,就相当于调用了n次call函数。即通过{h0,x1,x2,...,xn}直接得到{h1,h2,...,hn}
# 具体来说，设我们输入数据的格式为(batch_size, time_steps, input_size)；其中time_steps表示序列本书的长度,如
# Char RNN中，长度为10的句子对应的time_steps就等于10。最后的input_size就表示输入数据单个序列单个时间维度上固有的长度。


# inputs: shape = (batch_size, time_steps, input_size)
# cell:RNNCell
# initial_state: shape = (batch_size, cell.state_szie) 初始状态。一般可以取零矩阵
#x_data = np.float32(np.random.rand(32, 4, 100))
# input_content = tf.Variable(tf.ones([32,4,100]), dtype=np.float32),这一行有问题
#cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128) # state_size = 128
#inputs = tf.placeholder(np.float32, shape=(32, 4, 100)) # 32 是 batch_size
#initial_state = cell.zero_state(32, np.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
# 得到的outputs就是time_steps步所有的输出。它的形状为(batch_size, time_steps, cell.output_size)
# state是最后一步的隐状态，它的形状为(batch_size, cell.state_size)
#outputs, state = tf.nn.dynamic_rnn(cell, inputs,  initial_state=initial_state)

#init = tf.global_variables_initializer()

#with tf.Session() as sess:
#    sess.run(init)
#    print (sess.run(outputs, feed_dict={inputs: x_data}))
#    print (outputs.shape)


### 学习堆叠RNNCell:MultiRNNCell
# 将x输入第一层RNN的后得到隐层状态h, 这个隐层庄园就相当于第二层RNN的输入，第二层RNN的隐层状态
# 又相当于第三层RNN的输入，依次类推。可以使用tf.nn.rnn_cell.MultiRNNCell函数对RNNCell进行
# 堆叠

def get_a_cell():
    return tf.nn.rnn_cell.BasicRNNCell(num_units=128)
# 用tf.nn.rnn_cell MultiRNNCell创建3层RNN
cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell() for _ in range(3)])
# 得到的cell实际也是RNNCell的子类
# 它的state_size是(128,128,128)
# (128,128,128)并不是128*128*128的意思
# 而是表示共有3个隐层状态，每个隐层状态大小为128
print (cell.state_size)
inputs = tf.placeholder(np.float32, shape=(32, 100)) # 32是batch_size
h0 = cell.zero_state(32, np.float32)
output, h1 = cell.call(inputs, h0)
print (h1)











