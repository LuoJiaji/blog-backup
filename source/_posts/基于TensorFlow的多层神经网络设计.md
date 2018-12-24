---
title: 基于TensorFlow的多层神经网络设计
date: 2017-05-05 08:09:28
comments: false
tags:
- TensorFlow
- ML
- Python
---
今天利用TensorFlow来实现多层神经网络的设计，并用手写数字数据库对模型进行训练和测试。
<!--more-->
多层神经网络又叫多层感知机（Multi-layer Perception，MLP），多层神经网络包含输入层，输出层，隐藏层。隐藏层相当于模型的黑箱部分。理论上只要隐藏层包含的节点足够多，即使只有一个隐藏层的神经网络也可以拟合任意函数。隐藏层越多，越容易拟合复杂函数。理论研究表明，喂你喝复杂函数需要的隐藏层节点的数目，基本上随着隐藏层的数量增多成指数下降趋势。也就是说层数越多，神经网络所需要的隐含节点可以越少。

今天用到的模块和版本号：
Python版本：3.5.3
Tensorflow版本：1.0.1


首先导入数据
```Python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

然后就可以建立模型了
```Python
in_units = 784
h1_units = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# Define loss and optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
```
模型中，隐藏层的节点数设为300，在隐藏层中，用ReLU作为激活函数，这样可以防止多层神经网络中的梯度弥散。并在隐藏层中添加dropout层，Dropout可以用来防止过拟合。输出层用Softmax作为激活函数。

模型设计完成之后，就可以对模型参数进行初始化
```Python
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
```
最后就可以对模型进行训练，并验证识别的正确率。对模型进行3000次训练，训练过程中，每训练100次用测试数据对模型进行测试并输出测试结果
```
for i in range(3001):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
  if (i) % 100 == 0:
      correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      print(int((i) / 100), ':', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels,keep_prob: 1.0}))
```

最后得到输出图如下
![](/images/2017-5-5/2017-5-5-1.PNG)

可以看出，最终模型的正确率在98%左右，相比较之前的单层神经网络有了不小的提高。

参考资料：
* 《TensorFlow 实战》

