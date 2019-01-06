==========
算法
==========

.. contents:: 目录

包括哪些算法
===============

下面的算法已经在Spinning Up中实现：

- `Vanilla Policy Gradient`_ (VPG)
- `Trust Region Policy Optimization`_ (TRPO)
- `Proximal Policy Optimization`_ (PPO)
- `Deep Deterministic Policy Gradient`_ (DDPG)
- `Twin Delayed DDPG`_ (TD3)
- `Soft Actor-Critic`_ (SAC)

这些算法全部以 `MLP`_ （非循环） actor-critics 实现，使得他们能够非常适用于全观察、不基于图像的环境，例如 `Gym Mujoco`_ 环境。


.. _`Gym Mujoco`: https://gym.openai.com/envs/#mujoco
.. _`Vanilla Policy Gradient`: ../algorithms/vpg.html
.. _`Trust Region Policy Optimization`: ../algorithms/trpo.html
.. _`Proximal Policy Optimization`: ../algorithms/ppo.html
.. _`Deep Deterministic Policy Gradient`: ../algorithms/ddpg.html
.. _`Twin Delayed DDPG`: ../algorithms/td3.html
.. _`Soft Actor-Critic`: ../algorithms/sac.html
.. _`MLP`: https://en.wikipedia.org/wiki/Multilayer_perceptron


为什么是这些算法？
=====================

我们在这个包中选取了能够呈现强化学习近些年发展历程的核心算法。目前，在可靠性(stability)和采样效率(sample efficiency)这两个因素上表现最优的策略学习算法是PPO和SAC。从这些深度强化学习算法的设计和应用中，可以看出可靠性和采样效率的这两者的权衡。


On-Policy 算法
------------------------

Vanilla Policy Gradient(VPG) 是深度强化学习领域最基础、入门级的算法，远早于深度强化学习出现。VPG算法的核心部分可以追溯到上世纪80年代末、90年代初。之后，TRPO（2015）和PPO(2017)等更好的算法也相继诞生。。

这一系列工作的共同特征是这些算法都是On-Policy的，也就是说，他们不使用旧数据，因此他们在采用效率上表现相对较差。但这也是有原因的：这些算法直接优化我们关心的目标——策略表现。所以这个系列的算法都是用采样效率换取可靠性，之后的算法，从VPG到TRPO再到PPO，都在不断弥补采样效率方面的不足。

Off-Policy 算法
-------------------------

DDPG是一个和VPG同样重要的算法，尽管相对来说年轻很多。确定策略梯度（Deterministic Policy Gradients，DPG）理论，直到2014年才发表，而DDPG正是基于DPG提出的。DDPG和Q-learning算法很相似，同时学习Q函数和策略并通过更新相互提高。

DDPG和Q-Learning算法是 *off-policy* ，所以它们能够有效的重复利用旧数据——这一点是通过对贝尔曼方程（Bellman’s equations,也称动态规划方程）的优化实现，

但问题是，满足贝尔曼方程并不能保证一定有很好的策略性能。经验告诉我们，这样（满足贝尔曼方程）能取得很好的性能，同时采样效率很不错，但是由于缺乏保证，使得这类算法变得非常脆弱和不稳定。TD3和SAC是基于DDPG提出了很多新的方案来缓解这些问题。

代码格式
===========

所有Spinning Up的算法都按照固定的模板来实现。每个算法分为两个文件：一个算法文件，包括算法的核心逻辑，一个核心文件，包括各种运行算法所需的工具类。

算法文件
------------------

算法文件最开始是经验存储类的定义，作用是存储智能体和环境的互动信息。

接下来有一个运行算法，以及以下算法：
    
    1) Logger输出设定

    2) 随机种子的设定
    
    3) 环境实例化
    
    4) 为计算图创建placeholder
    
    5) 通过actor-critic函数传递算法函数

    6) 实例化经验缓存
    
    7) 损失函数和一些其他的函数
    
    8) 构建训练ops
    
    9) 构建TF Session并初始化参数
    
    10) 通过logger保存模型
    
    11) 定义运行算法主循环需要的函数（例如核心更新函数，获取行动函数，测试智能体函数等，取决于具体的算法）
    
    12) 运行算法主循环
    
        a) 让智能体在环境中开始运行
    
        b) 根据算法的主要方程式，周期性更新参数
    
        c) 打印核心性能数据并保存智能体

最后是从命令行读入设置的代码(ArgumentParser)。

核心文件
-------------

核心文件并没有像算法文件那样严格遵守模板，但也有一些相似的结构。

    1) 构建placeholder的函数

    2) actor-critic有关函数

    3) 其他有用的函数

    4) 与算法兼容的MLP actor-critic实现，策略和值函数都是通过简单的MLP来表示


