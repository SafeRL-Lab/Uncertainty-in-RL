# Uncertainty-in-RL







The repository is for Reinforcement-Learning Uncertainty research, in which we investigate various uncertain factors in RL, including single agent RL and multi-agent RL. 




***
The README is organized as follows:
- [1. Uncertainty in reward]
  * [1.1. Safe Single Agent RL benchmarks](#11-safe-single-agent-rl-benchmarks)
  * [1.2. Safe Multi-Agent RL benchmarks](#12-safe-multi-agent-rl-benchmarks)
- [2. Safe RL Baselines](#2-safe-rl-baselines)
  * [2.1. Safe Single Agent RL Baselines](#21-safe-single-agent-rl-baselines)
  * [2.2. Safe Multi-Agent RL Baselines](#22-safe-multi-agent-rl-baselines)
- [3. Surveys](#3-surveys)
- [4. Thesis](#4-thesis)
- [5. Book](#5-book)

***



### 1. Uncertainty in reward
#### 1.1. Inverse Reinforcement Learning 
- Apprenticeship Learning via Inverse Reinforcement Learning, [Paper](https://www.cs.utexas.edu/~sniekum/classes/RLFD-F15/papers/Abbeel04.pdf) (2004)
- Maximum Entropy Inverse Reinforcement Learning, [Paper](https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf?source=post_page---------------------------) (2008)
- Adversarial Inverse Reinforcement Learning, [Paper](https://arxiv.org/pdf/1710.11248.pdf) (2018)
- Inverse Reward Design, [Paper](https://proceedings.neurips.cc/paper/2017/file/32fdab6559cdfa4f167f8c31b9199643-Paper.pdf) (2017)

#### 1.2. Generative Adversarial Imitation Learning
- Generative Adversarial Imitation Learning, [Paper](https://proceedings.neurips.cc/paper/2016/file/cc7e2b878868cbae992d1fb743995d8f-Paper.pdf)(2016)
- A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models, [Paper](https://arxiv.org/pdf/1611.03852.pdf?source=post_page)(2016)
- Triple-GAIL: A Multi-Modal Imitation Learning Framework with Generative Adversarial Nets, [Paper](https://arxiv.org/pdf/2005.10622.pdf)(2020)

#### 1.3. Preference-based RL 
- Deep Reinforcement Learning from Human Preferences, [Paper](https://proceedings.neurips.cc/paper/2017/file/d5e2c0adad503c91f91df240d0cd4e49-Paper.pdf)(2017)
- Reward learning from human preferences and demonstrations in Atari, [Paper](https://proceedings.neurips.cc/paper/2018/file/8cbe9ce23f42628c98f80fa0fac8b19a-Paper.pdf)(2018)
- End-to-End Robotic Reinforcement Learning without Reward Engineering, [Paper](https://arxiv.org/pdf/1904.07854.pdf)(2019)
- PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training, [Paper](https://arxiv.org/pdf/2106.05091.pdf)(2021)
- Reward uncertainty for exploration in preference-based RL, [Paper](https://arxiv.org/pdf/2205.12401.pdf)(2022)

#### 1.4. Meta-Learning 
- MURAL: Meta-Learning Uncertainty-Aware Rewards for Outcome-Driven Reinforcement Learning, [Paper](http://proceedings.mlr.press/v139/li21g/li21g.pdf) (2021)




### 2. Uncertainty in transition

#### 2.1. Gaussian Process, Bayesian Neural Network 

- PILCO: A Model-Based and Data-Efficient Approach to Policy Search, [Paper](https://mlg.eng.cam.ac.uk/pub/pdf/DeiRas11.pdf)(2011)

- Improving PILCO with Bayesian Neural Network Dynamics Models，[Paper](http://mlg.eng.cam.ac.uk/yarin/website/PDFs/DeepPILCO.pdf)(2016)

- Weight Uncertainty in Neural Networks, [Paper](http://proceedings.mlr.press/v37/blundell15.pdf)(2015)

- Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning, [Paper](http://proceedings.mlr.press/v48/gal16.pdf)(2016)



#### 2.2. Model-Ensemble
 
- Deep Exploration via Bootstrapped DQN, [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9287440)(2016) 

- Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles, [Paper](https://proceedings.neurips.cc/paper/2017/file/9ef2ed4b7fd2c810847ffa5fa85bce38-Paper.pdf)(2017)

- Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models, [Paper](https://proceedings.neurips.cc/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf)(2018)
[Code](https://github.com/kchua/handful-of-trials)

- Model-Ensemble Trust-Region Policy Optimization,  [Paper](https://arxiv.org/pdf/1802.10592.pdf)(2018)
[Code](https://github.com/thanard/me-trpo.)


#### 2.3. Variational RL


- Auto-Encoding Variational Bayes, [Paper](https://arxiv.org/pdf/1312.6114.pdf?source=post_page---------------------------)(2013)

- Exploring State Transition Uncertainty in Variational Reinforcement Learning, [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9287440)(2020)



#### 2.4. Robust RL
- Robust Control of Markov Decision Processes with Uncertain Transition Matrices, [Paper](http://people.eecs.berkeley.edu/~elghaoui/Pubs/RobMDP_OR2005.pdf)(2005)

- Reinforcement Learning in Robust Markov Decision Processes, [Paper](https://proceedings.neurips.cc/paper/2013/file/0deb1c54814305ca9ad266f53bc82511-Paper.pdf)(2013)

- Robust Adversarial Reinforcement Learning, [Paper](http://proceedings.mlr.press/v70/pinto17a/pinto17a.pdf)(2017)

- Robust analysis of discounted Markov decision processes with uncertain transition probabilities, [Paper](http://www.amjcu.zju.edu.cn/amjcub/2020-2029/202004/417-436.pdf)(2020)

- Robust Multi-Agent Reinforcement Learning with Model Uncertainty, [Paper](https://proceedings.neurips.cc/paper/2020/file/774412967f19ea61d448977ad9749078-Paper.pdf)(2020)

- RAMBO-RL: Robust Adversarial Model-Based Offline Reinforcement Learning, [Paper](https://arxiv.org/pdf/2204.12581.pdf)(2022) [Code](https://github.com/marc-rigter/rambo)

### 3. Uncertainty in state

#### 3.1. 

- DESPOT: Online POMDP Planning with Regularization, [Paper](https://proceedings.neurips.cc/paper/2013/file/c2aee86157b4a40b78132f1e71a9e6f1-Paper.pdf)(2013)
- Bayesian Reinforcement Learning for Multiagent Systems with State Uncertainty, [Paper](https://d1wqtxts1xzle7.cloudfront.net/33866159/msdm2013-proceedings-with-cover-page-v2.pdf?Expires=1668977494&Signature=DjGUcOEoWzZOYs5hjOkhWSHzS9udXcua7ZwBvo1guJvdJmqkBT6I~IbVp8qWzUain9vcAdjAAJEZj5qvuAIsxXaWj7IFUcSUbI~aBM~5xGNrKtUpYLRSZuPNjVBR~QXQvZigdjiNGnkWCeJ-WtHzoxCA2UnLXmTyKid7UBPYaVzyMxjYlqTsEQjphgR04ePt522N0fHMBMjE3ofqX-SQLBdkzhe7M8QwuHhPH6FMWLZ4Kp1HyOaVz-KmZfdTFxfRyHXDxxw6ymqJq8jiqnXmNuFqkt8kB528JYpG2JJ0hhaWLdlur1xXzsrvzoC~KEsJU2KGvfInIPECHeJvnEukTA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA#page=80)(2013)

- Deep Recurrent Q-Learning for Partially Observable MDPs, [Paper](https://arxiv.org/pdf/1507.06527.pdf)(2015)

- Deep Reinforcement Learning with POMDP, [Paper](http://cs229.stanford.edu/proj2015/363_report.pdf)(2015)

- Intention-Aware Online POMDP Planning for Autonomous Driving in a Crows, [Paper](https://bigbird.comp.nus.edu.sg/m2ap/wordpress/wp-content/uploads/2016/01/icra15.pdf)(2015) 

- QMDP-Net: Deep Learning for Planning under Partial Observability, [Paper](https://proceedings.neurips.cc/paper/2017/file/e9412ee564384b987d086df32d4ce6b7-Paper.pdf)(2017)

- Deep Variational Reinforcement Learning for POMDPs, [Paper](http://proceedings.mlr.press/v80/igl18a/igl18a.pdf)(2018)

- On Improving Deep Reinforcement Learning for POMDPs, [Paper](https://arxiv.org/pdf/1704.07978.pdf)(2018) [Code](https://github.com/bit1029public/ADRQN)

- Shaping Belief States with Generative Environment Models for RL, [Paper](https://proceedings.neurips.cc/paper/2019/file/2c048d74b3410237704eb7f93a10c9d7-Paper.pdf)(2019)

- VARIATIONAL RECURRENT MODELS FOR SOLVING PARTIALLY OBSERVABLE CONTROL TASKS, [Paper](https://openreview.net/pdf?id=r1lL4a4tDB)(2020)  [Code](https://github.com/oist-cnru/Variational-Recurrent-Models)

- Particle Filter Recurrent Neural Networks, [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/5952)(2020)

- Recurrent Model-Free RL Can Be a Strong Baseline for Many POMDP, [Paper](https://arxiv.org/pdf/2110.05038.pdf)(2021) 

- Memory-based Deep Reinforcement Learning for POMDP, [Paper](https://arxiv.org/pdf/2102.12344.pdf)(2021) 

- Flow-based Recurrent Belief State Learning for POMDPs, [Paper](https://proceedings.mlr.press/v162/chen22q/chen22q.pdf)(2022) 

### 4. Uncertainty in observation
#### 4.1. 
- Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations, [Paper](https://arxiv.org/pdf/2003.08938.pdf)(2021)

- Incorporating Observation Uncertainty into Reinforcement Learning-Based Spacecraft Guidance Schemes, [Paper](https://arc.aiaa.org/doi/pdf/10.2514/6.2022-1765)(2022)
 





