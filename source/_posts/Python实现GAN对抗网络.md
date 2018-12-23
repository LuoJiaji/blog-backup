---
title: Python实现GAN对抗网络
date: 2017-10-28 19:18:22
comments: false
tags:
- GAN
- TensorFlow
- Python
---
GAN（对抗生成网络：Generative Adversarial Networks）是一类无监督学习的神经网络模型。
<!--more-->
生成对抗网络是目前一种非常受欢迎的网络模型，它最早在NIPS 2014 paper by Ian Goodfellow, et al中被提到，之后又出现了许多GAN的改进版本：DCGAN，Sequence-GAN, LSTM-GAN。

在GAN中第一个网路叫做生成网络$G(Z)$，第二个网络叫做鉴别网络$D(X)$

$$
\min \_G \max \_D V(D,G) =\mathbb E \_{x \sim p \_{data}(x)}[\log D(x)]  + \mathbb E \_{x \sim p \_x(x)}[log(1-D(G(z)))]
$$




GAN实现：
根据GAN的定义，需要两个网络模型。这可以是任何形式，可以是像卷积网络一样复杂的网络模型，也可以是简单的两层神经网络。这里使用两层的神经网络：
```python
# Discriminator Net
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

theta_D = [D_W1, D_W2, D_b1, D_b2]

# Generator Net
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

theta_G = [G_W1, G_W2, G_b1, G_b2]

def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob

def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit
```
代码中`generator(z)`输入100维的向量，并返回784维的向量，表示MNIST数据（28x28），`z`是$G(Z)$的先验。
`discriminator(x)`输入MNIST图片并返回表示真实MNIST图片的可能性。


然后声明GAN的训练过程。论文中的训练算法如下：
![](http://onaxllwtn.bkt.clouddn.com/2017-10-28-01.jpg)
代码实现如下：
```python
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))
```
损失函数加符号是因为公式来计算最大值，然而TensorFlow中优化器只能计算最小值。

然后根据上面的损失函数来训练对抗网络：
```python
# Only update D(X)'s parameters, so var_list = theta_D
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
# Only update G(X)'s parameters, so var_list = theta_G
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
```

最后得到结果：
![GAN training](http://onaxllwtn.bkt.clouddn.com/2017-10-28-02.gif)

参考资料：
* [Generative Adversarial Nets in TensorFlow](https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/  )