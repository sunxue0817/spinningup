============
介绍
============

.. contents:: 目录

这个项目是什么
============

欢迎来到深度强化学习（deep Reinforement Learning）的Spinning Up项目！这是一份由OpenAI提供的教育资源，旨在让深度强化学习的学习变得更加简单。

`强化学习`_ ，是一种通过教会智能体（agents）反复试错从而完成任务的机器学习方法。深度强化学习指的是强化学习和 `深度学习`_ 的结合。

这个模块包括一系列有用的资源，包括：

- 关于强化学习术语、算法和基础理论的简单 `介绍`_
- 关于如何成为强化学习的研究人员的 `文章`_
- 一个按照主题整理的重要论文 `清单`_
- 以及一些用来练手的 `项目`_




.. _`强化学习`: https://en.wikipedia.org/wiki/Reinforcement_learning
.. _`深度学习`: http://ufldl.stanford.edu/tutorial/

为什么创建这个项目
=================

我们最常听到的问题是：

    | 如果我想为AI安全做贡献，我应该如何开始？

在OpenAI，我们相信深度学习和深度强化学习会在未来的AI科技中扮演重要角色。为了确保AI的安全性，我们必须提出与此相契合的安全策略和算法。
因此我们鼓励每一个提出这个问题的人都来研究这些领域。

有很多帮助人们快速学习深度学习的资源，相比之下，深度强化学习显得门槛更高。首先，学生要有数学、编程和深度学习的背景。除此之外，他们需要
对于这一领域有高水准的理解：有哪些研究课题？为什么重要？哪些已经做出来了？，也需要认真的指导从而了解如何将算法的理论和代码
结合起来。

因为这个领域还很新，所以很难获得高水平的

One of the single most common questions that we hear is 

    | If I want to contribute to AI safety, how do I get started?

At OpenAI, we believe that deep learning generally---and deep reinforcement learning specifically---will play central roles in the development of powerful AI technology. To ensure that AI is safe, we have to come up with safety strategies and algorithms that are compatible with this paradigm. As a result, we encourage everyone who asks this question to study these fields.

However, while there are many resources to help people quickly ramp up on deep learning, deep reinforcement learning is more challenging to break into. To begin with, a student of deep RL needs to have some background in math, coding, and regular deep learning. Beyond that, they need both a high-level view of the field---an awareness of what topics are studied in it, why they matter, and what's been done already---and careful instruction on how to connect algorithm theory to algorithm code. 

The high-level view is hard to come by because of how new the field is. There is not yet a standard deep RL textbook, so most of the knowledge is locked up in either papers or lecture series, which can take a long time to parse and digest. And learning to implement deep RL algorithms is typically painful, because either 

- the paper that publishes an algorithm omits or inadvertently obscures key design details,
- or widely-public implementations of an algorithm are hard to read, hiding how the code lines up with the algorithm.

While fantastic repos like rllab_, Baselines_, and rllib_ make it easier for researchers who are already in the field to make progress, they build algorithms into frameworks in ways that involve many non-obvious choices and trade-offs, which makes them hard to learn from. Consequently, the field of deep RL has a pretty high barrier to entry---for new researchers as well as practitioners and hobbyists. 

So our package here is designed to serve as the missing middle step for people who are excited by deep RL, and would like to learn how to use it or make a contribution, but don't have a clear sense of what to study or how to transmute algorithms into code. We've tried to make this as helpful a launching point as possible.

That said, practitioners aren't the only people who can (or should) benefit from these materials. Solving AI safety will require people with a wide range of expertise and perspectives, and many relevant professions have no connection to engineering or computer science at all. Nonetheless, everyone involved will need to learn enough about the technology to make informed decisions, and several pieces of Spinning Up address that need. 



这个项目如何服务我们的使命
===========================

OpenAI's mission_ is to ensure the safe development of AGI and the broad distribution of benefits from AI more generally. Teaching tools like Spinning Up help us make progress on both of these objectives. 

To begin with, we move closer to broad distribution of benefits any time we help people understand what AI is and how it works. This empowers people to think critically about the many issues we anticipate will arise as AI becomes more sophisticated and important in our lives.

Also, critically, `we need people to help <https://jobs.lever.co/openai>`_ us work on making sure that AGI is safe. This requires a skill set which is currently in short supply because of how new the field is. We know that many people are interested in helping us, but don't know how---here is what you should study! If you can become an expert on this material, you can make a difference on AI safety.



代码设计的原则
======================

The algorithm implementations in the Spinning Up repo are designed to be 

    - as simple as possible while still being reasonably good, 
    - and highly-consistent with each other to expose fundamental similarities between algorithms.

They are almost completely self-contained, with virtually no common code shared between them (except for logging, saving, loading, and `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ utilities), so that an interested person can study each algorithm separately without having to dig through an endless chain of dependencies to see how something is done. The implementations are patterned so that they come as close to pseudocode as possible, to minimize the gap between theory and code. 

Importantly, they're all structured similarly, so if you clearly understand one, jumping into the next is painless. 

We tried to minimize the number of tricks used in each algorithm's implementation, and minimize the differences between otherwise-similar algorithms. To give some examples of removed tricks: we omit regularization_ terms present in the original Soft-Actor Critic code, as well as `observation normalization`_ from all algorithms. For an example of where we've removed differences between algorithms: our implementations of DDPG, TD3, and SAC all follow a convention laid out in the `original TD3 code`_, where all gradient descent updates are performed at the ends of episodes (instead of happening all throughout the episode). 

All algorithms are "reasonably good" in the sense that they achieve roughly the intended performance, but don't necessarily match the best reported results in the literature on every task. Consequently, be careful if using any of these implementations for scientific benchmarking comparisons. Details on each implementation's specific performance level can be found on our `benchmarks`_ page.


支持计划
============

We plan to support Spinning Up to ensure that it serves as a helpful resource for learning about deep reinforcement learning. The exact nature of long-term (multi-year) support for Spinning Up is yet to be determined, but in the short run, we commit to:

- High-bandwidth support for the first three weeks after release (Nov 8, 2018 to Nov 29, 2018).

    + We'll move quickly on bug-fixes, question-answering, and modifications to the docs to clear up ambiguities.
    + We'll work hard to streamline the user experience, in order to make it as easy as possible to self-study with Spinning Up. 

- Approximately six months after release (in April 2019), we'll do a serious review of the state of the package based on feedback we receive from the community, and announce any plans for future modification, including a long-term roadmap.

Additionally, as discussed in the blog post, we are using Spinning Up in the curriculum for our upcoming cohorts of Scholars_ and Fellows_. Any changes and updates we make for their benefit will immediately become public as well.


.. _`介绍`: ../spinningup/rl_intro.html
.. _`文章`: ../spinningup/spinningup.html
.. _`Spinning Up essay`: ../spinningup/spinningup.html
.. _`清单`: ../spinningup/keypapers.html
.. _`code repo`: https://github.com/openai/spinningup
.. _`项目`: ../spinningup/exercises.html
.. _`rllab`: https://github.com/rll/rllab
.. _`Baselines`: https://github.com/openai/baselines
.. _`rllib`: https://github.com/ray-project/ray/tree/master/python/ray/rllib
.. _`mission`: https://blog.openai.com/openai-charter/
.. _`regularization`: https://github.com/haarnoja/sac/blob/108a4229be6f040360fcca983113df9c4ac23a6a/sac/distributions/normal.py#L69
.. _`observation normalization`: https://github.com/openai/baselines/blob/28aca637d0f13f4415cc5ebb778144154cff3110/baselines/run.py#L131
.. _`original TD3 code`: https://github.com/sfujim/TD3/blob/25dfc0a6562c54ae5575fad5b8f08bc9d5c4e26c/main.py#L89
.. _`benchmarks`: ../spinningup/bench.html
.. _Scholars : https://jobs.lever.co/openai/cf6de4ed-4afd-4ace-9273-8842c003c842
.. _Fellows : https://jobs.lever.co/openai/c9ba3f64-2419-4ff9-b81d-0526ae059f57


