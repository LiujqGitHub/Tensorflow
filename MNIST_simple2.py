import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 载入数据集
# 网上下载数据集
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少个批次,//整除
n_batch = mnist.train.num_examples // batch_size

# 定义两个placeholder,x:训练集，y:训练集对应标签
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 输入层
W_L1 = tf.Variable(tf.random_normal([784,2000]))
b_L1 = tf.Variable(tf.random_normal([1, 2000]))
Wx_plus_b_L1 = tf.matmul(x, W_L1) + b_L1
L1 = tf.nn.tanh(Wx_plus_b_L1)

# 输出层
W_L2 = tf.Variable(tf.zeros([2000,10]))
b_L2 = tf.Variable(tf.zeros([1,10]))
Wx_plus_b_L2 = tf.matmul(L1, W_L2) + b_L2
prediction = tf.nn.softmax(Wx_plus_b_L2)

# 定义二次代价函数，reuce_mean；取平均值，squre:x*x
loss = tf.reduce_mean(tf.square(y-prediction))
# 使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#初始化变量
init = tf.global_variables_initializer()
#argmax返回一维张量中最大的值所在的位置,1:行向量,默认：列;结果存放在一个布尔型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#求准确率，tf.cast转换为32float,True:1.0,False:0.0,tf.recude_mean，求和取平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21): # 所有数据过了21次
        for batch in range(n_batch):# 执行完一次该循环==过了一遍所有数据
            # data,label
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs, y:batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))



