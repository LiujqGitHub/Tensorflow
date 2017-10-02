import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy生成200个随机点
# 得到200个点，一维数据
# x_data = np.linspace(-0.5, 0.5, 200)
# 得到200行，一列的数据
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
# 得到一个大致为U的形状，但比较混乱
y_data = np.square(x_data) + noise

# 定义两个placeholder，根据样本定义的
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络的中间层
# 输入层是一个数（一列），故输入层使用一个神经元（如果两个数，即两个神经元，两列），隐层神经元个数可自由设置
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))
bias_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L1 = tf.matmul(x,Weights_L1) + bias_L1
L1 = tf.nn.tanh(Wx_plus_b_L1) # 激活函数

# 定于输出层,输出是一个数，故定义一个神经元
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
bias_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + bias_L2
prediction = tf.nn.tanh(Wx_plus_b_L2) # 激活函数

# 定义代价函数以及训练的方法
# 真实值-预测值 平方求平均
# y是知道的，prediction需要用到输出层信号的总和，需要优化隐层的权值与bias，输出层的权值和bias
loss = tf.reduce_mean(tf.square(y-prediction))
# 使用梯度下降法训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(train_step,feed_dict={x:x_data, y:y_data})
    # 获得预测值,进行预测
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图；散点图打印样本点；r:red,-:实线
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-',lw=5)
    plt.show()