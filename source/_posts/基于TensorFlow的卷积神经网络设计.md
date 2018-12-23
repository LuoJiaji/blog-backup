---
title: 基于TensorFlow的卷积神经网络设计
date: 2017-05-12 21:08:37
comments: false
tags:
- TensorFlow
- ML
- Python
---
今天继续来写一篇关于TensorFlow的，用TensorFlow来设计一个卷积神经网络，并用于手写体识别。
<!--more-->
卷积神经网络(Convolutional Neural Network,CNN)，应该算是一种比较经典的网络模型，很多深度学习的框架中都可以看到CNN的身影。因此今天就用TensorFlow来实现一个卷积神经网络。

今天用到的模块和版本号：
Python版本：3.5.3
Tensorflow版本：1.0.1

第一步跟之前的一样，先导入数据：
```Python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```

然后定义一下网络节点的权重和偏置
```Python
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
```

然后来定义卷基层和池化层
```Python
def conv2d(x,W):
    # stride [1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') # must have strides[3] = 1

def max_pool_2X2(x):
    # stride [1,x_movement,y_movement,1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
```
`tf.nn.conv2d()`是卷积层其中`x`为输入，`W`为卷积参数，`strides`表示卷积模板的移动步长，都是1代表会不遗漏地划过图片的每一个点。`padding`代表边界的处理方式，这里的SAME参数代表给边界加上`padding`让卷积的输入和输出保持相同的尺寸。
`tf.nn.max_pool()`是池化层，这里使用2\*2的最大池化，既将一个2\*2的像素块将为1\*1的像素块。最大池化会保留像素块中灰度值最高的一个像素。

然后来定义网络模型
```Python
xs = tf.placeholder(tf.float32, [None,784]) # 28*28
ys = tf.placeholder(tf.float32, [None,10])
keep_prop = tf.placeholder(tf.float32)

x_image = tf.reshape(xs,[-1,28,28,1])
# print(x_image.shape) #[n_sample,28,28,1]

## convl layer ##
W_conv1 = weight_variable([5,5,1,32])  # patch:5*5, in size:1, outsize:32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size:28*28*32
h_pool1 = max_pool_2X2(h_conv1) # output size: 14*14*32

## conv2 layer ##
W_conv2 = weight_variable([5,5,32,64])  # patch:5*5, in size:32, out size:64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size:14*14*64
h_pool2 = max_pool_2X2(h_conv2)                           # output size: 7*7*64

##func1
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])#n_samples,7,7,64 -> n_samples,7*7*64
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prop)

##func2
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+ b_fc2)


cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))  # loss

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
```

在CNN网络模型中包含两个卷基层和两个全连接层，用ReLU作为激活函数，这样可以避免多层神经网络的梯度弥散和梯度爆炸。在每层之间再加上一个dropout层，这是为了防止模型过拟合。用cross entropy作为损失函数，优化器使用Adam学习速率设为1e-4


然后定义一个用来计算网络模型识别正确率的函数
```Python
def compute_accuracy(v_xs , v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prop: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prop: 1})
    return result
```


最后对模型进行训练，并通过`compute_accuracy()`来验证模型的正确率，并打印出正确率
```Python
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs: batch_xs, ys: batch_ys, keep_prop : 0.5})
    if i % 100 == 0:
        print(i/100,':',compute_accuracy(mnist.test.images, mnist.test.labels))

```
最后可以得到，正确率为98.4% ，如果增加迭代的次数，模型的正确率还能再高一点，大约在99%左右。


参考资料:
* 《TensorFlow实战》
