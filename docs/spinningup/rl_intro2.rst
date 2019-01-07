==============================
第一部分：各种各种的强化学习算法
==============================

.. contents:: 目录
    :depth: 2

我们已经介绍了深度学习的基础术语和符号，现在可以讨论更有丰富的资料：现代强化学习的整体发展和算法设计时候要考虑的各种因素之间的权衡。

强化学习算法的分类
===========================

.. figure:: ../images/rl_algorithms_9_15.svg
    :align: center

    一个不是很详细但是十分有用的现代强化学习算法分类。  `引用自`_

要先声明的是：很难准确、全面的把所有现代强化学习算法列出来，因为这些算法并不能很好的用树形结构表示。同时，要把这么多内容放在一篇文章里并且要求便于理解消化，我们必须省略掉一些更加先进的资料，例如探索（exploration），迁移学习（transfer learning），元学习（meta learning）等。也就是说，我们的目标是：

* 只强调深度强化学习中，关于学习什么和怎么学习的最基础的设计选择
* to expose the trade-offs in those choices,
* and to place a few prominent modern algorithms into context with respect to those choices.

免模型学习（Model-Free） VS 有模型学习（Model-Based）
----------------------------

强化学习算法最重要的区分点之一就是 **智能体是否能完整获得或学习到所在环境的模型**。 环境的模型，指的是一个预测状态转换和奖励的函数。

有模型学习最大的优势在于智能体能够 **看的更远从而做出计划**，走到每一步的时候，都提前尝试未来可能的选择，然后很清晰地从这些候选项中进行选择。

最大的缺点就是智能体往往不能获得环境的真实模型。如果智能体想在一个场景下使用模型，那它必须完全从经验中学习，这就带来很多挑战。最大的挑战就是，智能体探索出来的模型和真实模型之间的误差，会导致智能体在已学习到的模型中表现很好，但是在真实的环境中表现得不是最优（甚至很差）。从根本上来，基于模型的学习是非常困难的，即便是你愿意花费大量的时间和计算力，最终的结果也有可能达不到预期的效果。

要学习什么
-------------

另一个强化学习算法重要的区分点是 **要学习什么**。常提到的主题包括：

* 策略，不管是随机的还是确定性的
* 行动奖励函数（Q函数）
* 值函数
* 环境模型

免模型学习中要学习什么
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

有两种用来表示和训练免模型学习强化学习算法的方式：

**策略优化（Policy Optimization）**：这个系列的方法可以表示为： :math:`\pi_{\theta}(a|s)` 。 它们直接对目标函数进行梯度下降进行优化，或者间接地，对目标函数的局部近似函数进行优化。优化基本都是基于 **同策路**的，也就是说每一步更新只会用到最新的策略执行时采集到的数据。策略优化也包括学习出 :math:`V_{\phi}(s)`，作为 :math:`V^{\pi}(s)`的近似，从而知道如何更新策略。

基于策略优化的方法举例：

* `A2C / A3C`_, 通过梯度下降直接最大化性能
* `PPO`_, 不直接通过最大化性能更新，而是最大化 *目标估计* 函数，这个函数是对目标函数 :math:`J(\pi_{\theta})`的近似估计。

**Q-Learning** 
.. math::
    
    a(s) = \arg \max_a Q_{\theta}(s,a).

基于Q-Learning优化的方法

* `DQN`_, 一个让深度强化学习得到发展的经典方法
* and `C51`_, a variant that learns a distribution over return whose expectation is :math:`Q^*`.
* 以及 `C51`_, 学习一个关于回报的分布函数，其期望是 :math:`Q^*` 

**策略优化和Q-Learning的比较** 策略优化的主要优势是这类方法很稳定很可靠，从直觉上将，你是直接在优化你想要的东西。Q-learning方法通过训练 :math:`Q_{\theta}` 以满足自我一致，间接地优化智能体的表现。这种方法有很多失败的情况，所以相对来说没有那么稳定。[1]_ 但是，Q-learning有效的时候能获得更好的采样效率，因为它们能够比策略优化更加有效地重用数据。

**策略优化和Q-learning的融合方法** 意外的是，策略优化和Q-learning并不是不能兼容的（在某些场景下，它们两者是 `等价的`_ ），也有很多算法介于两种极端之间。这个范围之类的算法能够很好的平衡好两者之间的优点和缺点，比如说：

* `DDPG`_ ，同时学习确定性策略和Q-函数，并通过彼此互相提升
* `SAC`_ ，使用随机策略、熵正则化和一些其它技巧来稳定学习并在标准基准上得分高于DDPG的变种

.. [1] 关于更多Q-learning方法如何失败的，参见： 1) 经典论文 `Tsitsiklis and van Roy`_, 2) 最近的文章 `review by Szepesvari`_ (在 4.3.2章节)  3) `Sutton and Barto`_ 的第11章节，尤其是 11.3 (on "the deadly triad" of function approximation, bootstrapping, and off-policy data, together causing instability in value-learning algorithms).


.. _`Bellman equation`: ../spinningup/rl_intro.html#bellman-equations
.. _`Tsitsiklis and van Roy`: http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf
.. _`review by Szepesvari`: https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf
.. _`Sutton and Barto`: https://drive.google.com/file/d/1xeUDVGWGUUv1-ccUMAZHJLej2C7aAFWY/view
.. _`equivalent`: https://arxiv.org/abs/1704.06440

有模型强化学习要学习什么
-------------------------------

不同于免模型学习，有模型学习方法不是很好分类：很多方法之间都会有交叉。我们会列举一些例子，当然肯定不够详尽，覆盖不到全部。在这些例子里面， **模型**有么已知要么可以学习到。

**背景：纯规划** 这个最基础的方法，从不用表示策略，而是纯使用计划技术来选择行动，比如 `model-predictive control`_ (MPC)。在MPC中，智能体每次观察环境的时候，都会计算出一个对于当前模型最优的计划，这里的计划指的是未来一个固定时间段内，智能体会采取的行动。（超过视野的未来奖励可以通过）
**Background: Pure Planning.** The most basic approach *never* explicitly represents the policy, and instead, uses pure planning techniques like `model-predictive control`_ (MPC) to select actions. In MPC, each time the agent observes the environment, it computes a plan which is optimal with respect to the model, where the plan describes all actions to take over some fixed window of time after the present. (Future rewards beyond the horizon may be considered by the planning algorithm through the use of a learned value function.) The agent then executes the first action of the plan, and immediately discards the rest of it. It computes a new plan each time it prepares to interact with the environment, to avoid using an action from a plan with a shorter-than-desired planning horizon.

* The `MBMF`_ work explores MPC with learned environment models on some standard benchmark tasks for deep RL.

**Expert Iteration.** A straightforward follow-on to pure planning involves using and learning an explicit representation of the policy, :math:`\pi_{\theta}(a|s)`. The agent uses a planning algorithm (like Monte Carlo Tree Search) in the model, generating candidate actions for the plan by sampling from its current policy. The planning algorithm produces an action which is better than what the policy alone would have produced, hence it is an "expert" relative to the policy. The policy is afterwards updated to produce an action more like the planning algorithm's output.

* The `ExIt`_ algorithm uses this approach to train deep neural networks to play Hex.
* `AlphaZero`_ is another example of this approach.

**Data Augmentation for Model-Free Methods.** Use a model-free RL algorithm to train a policy or Q-function, but either 1) augment real experiences with fictitious ones in updating the agent, or 2) use *only* fictitous experience for updating the agent. 

* See `MBVE`_ for an example of augmenting real experiences with fictitious ones.
* See `World Models`_ for an example of using purely fictitious experience to train the agent, which they call "training in the dream."

**Embedding Planning Loops into Policies.** Another approach embeds the planning procedure directly into a policy as a subroutine---so that complete plans become side information for the policy---while training the output of the policy with any standard model-free algorithm. The key concept is that in this framework, the policy can learn to choose how and when to use the plans. This makes model bias less of a problem, because if the model is bad for planning in some states, the policy can simply learn to ignore it.

* See `I2A`_ for an example of agents being endowed with this style of imagination.

.. _`model-predictive control`: https://en.wikipedia.org/wiki/Model_predictive_control
.. _`ExIt`: https://arxiv.org/abs/1705.08439
.. _`World Models`: https://worldmodels.github.io/



分类中提到的算法链接
===============================

.. _`引用自`: 

.. [#] `A2C / A3C <https://arxiv.org/abs/1602.01783>`_ (Asynchronous Advantage Actor-Critic): Mnih et al, 2016
.. [#] `PPO <https://arxiv.org/abs/1707.06347>`_ (Proximal Policy Optimization): Schulman et al, 2017 
.. [#] `TRPO <https://arxiv.org/abs/1502.05477>`_ (Trust Region Policy Optimization): Schulman et al, 2015
.. [#] `DDPG <https://arxiv.org/abs/1509.02971>`_ (Deep Deterministic Policy Gradient): Lillicrap et al, 2015
.. [#] `TD3 <https://arxiv.org/abs/1802.09477>`_ (Twin Delayed DDPG): Fujimoto et al, 2018
.. [#] `SAC <https://arxiv.org/abs/1801.01290>`_ (Soft Actor-Critic): Haarnoja et al, 2018
.. [#] `DQN <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_ (Deep Q-Networks): Mnih et al, 2013
.. [#] `C51 <https://arxiv.org/abs/1707.06887>`_ (Categorical 51-Atom DQN): Bellemare et al, 2017
.. [#] `QR-DQN <https://arxiv.org/abs/1710.10044>`_ (Quantile Regression DQN): Dabney et al, 2017
.. [#] `HER <https://arxiv.org/abs/1707.01495>`_ (Hindsight Experience Replay): Andrychowicz et al, 2017
.. [#] `World Models`_: Ha and Schmidhuber, 2018
.. [#] `I2A <https://arxiv.org/abs/1707.06203>`_ (Imagination-Augmented Agents): Weber et al, 2017 
.. [#] `MBMF <https://sites.google.com/view/mbmf>`_ (Model-Based RL with Model-Free Fine-Tuning): Nagabandi et al, 2017 
.. [#] `MBVE <https://arxiv.org/abs/1803.00101>`_ (Model-Based Value Expansion): Feinberg et al, 2018
.. [#] `AlphaZero <https://arxiv.org/abs/1712.01815>`_: Silver et al, 2017 


