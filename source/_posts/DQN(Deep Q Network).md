---
title: DQN(Deep Q Network)
date: 2017-05-25 19:03:43
comments: false
tags:
- RL
---

DQN是Deep Q Network的简称，是一种将强化学习的方法(Q-Learning)和神经网络(Neural Networks)相结合的一种新的算法。
<!--more-->
Q-learning算法是1989年Watkins提出来的一种强化学习的算法，在次基础上2015年nature的论文上提出了在Q-Learning基础上改进的算法DQN。
Q-Learning中需要一个Q表来存储每个状态中不同动作对应的值函数。这种算法在有限状态中会比较有效，但当状态空间变得非常大或者连续的情况下，Q-Learning的算法就需要一个非常大的Q表来存储不同的状态，这样使得算法变得难以实现。因此结合神经网络的方法出现了DQN算法。
DQN可以通过神经网络来拟合状态值，这样就避免了Q-Learning中需要存储大量状态空间的问题。

下面就是DQN的算法更新过程
![DQN算法的更新过程](/images/2017-5-25/2017-5-25-1.JPG)

第1行，初始化状态存储空间$D$，存储空间容量为$N$

第2行，用随机权值 $\theta$ 初始化动作值函数的神经网络$Q$

第3行，初始化目标动作值函数$\hat Q$，并且网络权值$\theta^-=\theta$ 

第4行，循环每一次事件

第5行，初始化第一个状态$s_1$ ，通过预处理得到对应的状态特征 $\phi\_1=\phi(s)$ (用来作为神经网络的输入)

第6行，循环每一步动作

第7行，利用$\epsilon$随机选择一个动作$a_t$

第8行，如果$\epsilon$概率没有发生，则选择对应状态中所有动作的最大值$a_t=\arg \max_a Q(\phi(s_t),a;\theta)$

第9行，执行动作$a\_t$，得到奖励值$r\_t$和推测的下一步$x\_{t+1}$

第10行，令$s\_{t+1} = s\_t,a\_t,x\_{t+1}$，并且对状态$s\_{t+1}$做预处理，$\phi\_{t+1}=\phi(s\_{t+1})$

第11行，将转移过程$(\phi\_t,a\_t,r\_t,\phi\_{t+1})$保存在存储空间$D$中

第12行，从存储空间$D$中，随机采样一组训练数据$(\phi\_j,a\_j,r\_j,\phi\_{j+1})$

第13行，判断事件是否在$j+1$步终止，如果终止，则$y\_j = r\_j$；否则$ y\_j = r\_j + \gamma max\_{a^\`} \hat Q (\phi\_{j+1}+a^\`;\theta^-)$

第14~15行，通过梯度下降$(y\_t - Q(\phi\_j,a\_j;\theta) )^2$来优化参数$\theta$

第17行，每经过$C$步，将动作值函数的预测网络赋值给目标动作值网络网络 $\hat Q = Q $



从上面的算法过程可以看出DQN算法在原有Q-Learning的基础上做了一些改进

首先，DQN利用神将网络来通过状态计算相应的动作值，这样就可以避免在状态空间较大或者状态空间为连续的情况下Q-Learning算法难以实现的问题。

其次，算法通过存储一定数量的过往经历，并在这些经历中随机选择一些进行重复学习(算法第12行)，来对神经网络进行训练。这样通过经验进行回放学习的思想来自于人类大脑中的海马体，海马体是人类大脑中负责记忆和学习的部分，当人在睡觉的时候，海马体会把一天的记忆重放给大脑皮层。

最后，通过独立设置目标网络来进行计算目标动作值，这样有利于网络的收敛，这个思想就像Q-Learning与Sarsa的区别一样，Sarsa的动作值更新过程和行为选择过程是相互关联的，这样当状态过多时容易导致算法难以收敛，而Q-Learning的动作值更新和动作选择是两个相对独立的过程，这样可以使算法更容易收敛，同时，在对存储的状态行为进行随机采样时也可以打乱行为之间的联系，也是促进算法收敛的一种方法(算法第12行)。





参考资料：
* [深度强化学习系列 第一讲 DQN](https://zhuanlan.zhihu.com/p/26052182)
* [DQN从入门到放弃5 深度解读DQN算法](https://zhuanlan.zhihu.com/p/21421729)
* [DQN 算法更新 (Tensorflow)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-1-DQN1/)
* [DQN 思维决策 (Tensorflow)](https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/)

