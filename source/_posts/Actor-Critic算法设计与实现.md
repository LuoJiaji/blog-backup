---
title: Actor Critic算法设计与实现
date: 2017-08-25 20:14:53
tags:
- RL
- TensorFlow
- Python
---

今天介绍强化学习中的一种新的算法，叫做Actor Critic。
<!--more-->
## 介绍
Actor Critic将Policy Gradient（Actor部分）和Function Approximation（Critic部分）相结合，`Actor`根据当前状态对应行为的概率选择行为，`Critic`通过当前状态，奖励值以及下一个状态进行学习同时返回`Actor`当前行为评分，`Actor`根据`Critic`给出的评分对行为概率进行修正。

Actor Critic的优缺点：
* **优点**：Actor Critic不用想DQN或者Policy Gradient那样需要对探索的状态，动作和奖励信息进行存储。而是可以进行单步学习和更新，这样学习效率要比DQN和Policy Gradient快。
* **缺点**：由于单步更新参照的信息有限，而且`Actor`和`Critic`要同时学习，因此学习比较难以收敛。


![](/images/2017-8-25/2017-8-25-1.png)


## 代码实现
下面介绍Actor Critic的具体实现过程。
用到的Python库：
Python：3.5.3
TensorFlow：1.0.1
gym：0.8.1

试验环境依然使用`gym`中的`CartPole-v0`。

首先介绍Actor部分的代码。
```python
class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})   # get probabilities for all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
```

代码中首先传递环境的状态数量`n_features`，动作数量`n_actions`和学习效率`lr`。然后构建用于`Actor`学习的神经网络，网络包含一个含有20个节点的隐藏层。
`Actor.learn()`通过当前状态`s`，动作`a`和`Critic`给出的时间差分误差`td`进行学习。
`Actor.Choose_action()`通过当前状态`s`计算出每个动作的概率，然后根据相应的概率来选择动作。


下面是`Critic`部分的代码：
```python
class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error
```
代码中建立一个输入层节点为`s`,隐藏层节点为20，输出层节点为1的神经网络。
`Critic.learn()`根据当前状态`s`,奖励值`r`和下一步的状态`s_`进行学习，并返回时间差分误差`td`给`actor.learn()`

然后是完整的训练过程：
```python
import gym
from ActorCritic import *

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# Superparameters

OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 100  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

env = gym.make('CartPole-v0')
env.seed(1)  # reproducible

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

sess = tf.Session()

actor = Actor(sess, n_features=N_F, n_actions=N_A, lr=LR_A)
critic = Critic(sess, n_features=N_F, lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor

sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while True:
        if RENDER: env.render()

        a = actor.choose_action(s)

        s_, r, done, info = env.step(a)

        if done: r = -20

        track_r.append(r)

        td_error = critic.learn(s, r, s_)  # gradient = grad[r + gamma * V(s_) - V(s)]
        actor.learn(s, a, td_error)     # true_gradient = grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))
            break
```

通过训练发下，通过3000次迭代，Actor Critic算法的收敛性确实不是太理想，没有DQN和Policy Gradient的效果好。

通过TensorBoard可以查看网络的结构如下：
```
Tensorboard --logdir logs
```
![](/images/2017-8-25/2017-8-25-2.png)


参考资料：
* [Actor Critic (Tensorflow)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/6-1-actor-critic/ )
