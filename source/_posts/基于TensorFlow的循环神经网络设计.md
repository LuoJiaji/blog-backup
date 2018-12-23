---
title: 基于TensorFlow的循环神经网络设计
date: 2017-06-08 17:38:54
comments: false
tags:
- TensorFlow
- ML
- Python
---
循环神经网络(RNN)是区别于卷积神经网络的一种网络结构。适用于自然语言处理，文本分析，机器翻译等领域。
<!--more-->
循环神经网络出现于20世纪80年代，但是早期应用有限，随着神经网络结构的进步和硬件的支持，RNN变得越来越流行。
循环神经网络对时间序列数据比较有效，
今天用TensorFlow实现一个循环神经网络，数据依旧使用手写体数据。

今天用到的模块和版本号：
Python版本：3.5.3
Tensorflow版本：1.0.1

第一步，导入数据
```Python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

然后定义一些参数
```Python
# 超参数
lr = 0.001                  # 学习率
training_iters = 1000       # 迭代次数
batch_size = 128            # 批大小

n_inputs = 28   # 输入数据大小
n_steps = 28    # 步长
n_hidden_units = 128   # 隐藏层节点数
n_classes = 10      # 分类数量
```

然后定义神经网络中的权重和偏置
```Python
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}
```
代码中权重初始化为服从正态分布的随机数，偏置初始化为常数0.1

然后定义神经网络的输入和输出
```Python 
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])
```
用占位符定义循环神经网络的输入和标签输出，这个输出并不是神经网络的输出值，而是数据的标签值，计算结果直接由神经网络输出。

然后就到了关键部分，定义循环神经网络：
```Python
def RNN(X, weights, biases):
    # hidden layer for input to cell

    # transpose the inputs shape from
    # X ==> (128 batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    # X_in = (128 batch * 28 steps, 128 hidden)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    # X_in ==> (128 batch, 28 steps, 128 hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    # basic LSTM Cell.
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)

    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results
```
代码中`tf.reshape()`用来改变数据的维度，将(128,28,28)的三维向量改为(128\*28,28)的二维向量。然后经过输入层之后再将数据转换三维的。`tf.contrib.rnn.BasicLSTMCell()`调用LSTM循环神经网络单元。然后用`tf.nn.dynamic_rnn()`创建一个由循环神经网络单元组成的循环神经网络。`tf.transpose()`用来对矩阵进行转换。`tf.unstack()`用来将张量数据解开。最后用`tf.matmul()`计算输出层的结果。

然后调用设计好的神经网络并计算损失函数和优化方法，然后计算预测输出和准确率：
```Python
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
```

最后对模型进行训练，每隔20步计算并打印模型的识别效率。
```Python
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12

    init = tf.global_variables_initializer()
    sess.run(init)
    step = 0
    while step  < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(step,sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1
```
最终模型的识别率可以达到97%左右。


参考资料：
* 《TensorFlow实战》
* [循环神经网络(RNN, Recurrent Neural Networks)介绍](http://blog.csdn.net/heyongluoyao8/article/details/48636251) 
* [RNN LSTM 循环神经网络 (分类例子)](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-08-RNN2/)