# AIDD

This repository will contain the official PyTorch implementation of:

**Automated Discovery of Interactions and Dynamics for Large Networked Dynamical Systems**

Yan Zhang1, Yu Guo2,3, Zhang Zhang1, Mengyuan Chen1, Shuo Wang1, and Jiang Zhang1,*

1Beijing Normal University, Beijing, China

2State Key Laboratory for Novel Software Technology at Nanjing University, Nanjing, China

3Software Institute, Nanjing University, Nanjing, China

*zhangjiang@bnu.edu.cn

https://arxiv.org/abs/2101.00179

## ABSTRACT

Understanding the mechanisms of complex systems is very important. According to specific dynamic rules, a networked dynamical system, understanding a system as a group of nodes interacting on a given network, is a powerful tool for modeling complex systems. However, finding such models according to the time series of behaviors is hard. Conventional methods
can work well only on small networks and some types of dynamics. This paper proposes a unified framework for Automated Interaction network and Dynamics Discovery (AIDD) on various network structures and dynamics based on a stochastic gradient descent algorithm. The experiments show that AIDD can be applied to large systems with thousands of nodes and
robust against noise and missing information. We further propose a new method to test data-driven models based on control experiments. The results show that AIDD has learned the real network dynamics correctly.

## Requirements

- python 3.7.7
- pytorch 1.5.0
- networkx 2.4
- netrd 0.2.2
- numpy 1.18.1
- scipy 1.4.1

## Directory description

### AIDD

We put the single step prediction training algorithm of AIDD in this folder.

The framework consists of two parts: a network generator and a dynamics learner. The input of the model is the state information of all nodes at time t, and the output of the model is the predicted state information of all nodes at time t +1. The inferred adjacency matrix \hat{A} can also be retrieved from the network generator.

#### Data Generation

The files whose name starts with "generate" are files that generate data.For example, if you want to generate data of spring,you can run the file generate_spring.py

```python
python generate_spring.py
or
nohup python -u generate_spring.py > generate_spring.txt 2>&1 & #Save the output file to a text file
```

#### Run Experiment

The files whose name starts with "train" are files that train. For example, if you want to run the experiment of spring, you can run the file train_spring.py

```python
python train_spring.py
or
nohup python -u train_spring.py > train_spring.txt 2>&1 & #Save the output file to a text file
```

#### Test Model Performance

The files whose name starts with "test" are files that test. For example, if you want to test the model of spring, you can run the file test_spring.py

```python
python test_spring.py
or
nohup python -u test_spring.py > test_spring.txt 2>&1 & #Save the output file to a text file
```



### AIDD_Multi_step_prediction

We put the multi-step prediction training algorithm of AIDD in this folder.

In continuous dynamics prediction tasks, to obtain a high prediction accuracy, multi-step prediction training is needed. That is, input the current state at time t, to predict the states at time t +1; t+2;...; t +T.

#### Data Generation

you can run the file generate_spring.py

```python
python generate_spring.py
or
nohup python -u generate_spring.py > generate_spring.txt 2>&1 & #Save the output file to a text file
```

#### Run Experiment

you can run the file train_spring_mutistep.py

```python
python train_spring_multistep.py
or
nohup python -u train_spring_multistep.py > train_spring_multistep.txt 2>&1 & #Save the output file to a text file
```

#### Test Model Performance

you can run the file test_spring_multistep.py

```python
python test_spring_multistep.py
or
nohup python -u test_spring_multistep.py > test_spring_multistep.txt 2>&1 & #Save the output file to a text file
```



### Controller_Optimization

We design control experiments to test a learned model. The control experiment can be separated into two phases. In the first stage, we find the optimized controller’s parameters on the learned network dynamics to achieve the designed objective. In the second stage, we do the same optimization but directly on the ground truth model. After that, we compare the results on controls.

Before you run the controller experiment , you must run the single step  prediction training algorithm of AIDD to get the learned model. And you should put the learned model into  the   "model" directory under the current folder, put the corresponding data into the "data" directory under the current folder.

For convenience, we put our trained model and data in the corresponding folder, you can use it directly.

#### Optimized controller's parameters on the learned model

For example, if you want to control the model of spring, you can run the file control_spring_ourmodel_ournet.py

```python
python control_spring_ourmodel_ournet.py
```

####  Do the same optimization on the ground truth model

For example, if you want to control the model of spring, you can run the file control_spring_controlmodel_realdyn.py

```python
python control_spring_controlmodel_realdyn.py
```

#### Optimized controller's parameters on the real dynamics system

For example, if you want to control the model of spring, you can run the file control_spring_realdyn_realnet.py

```python
python control_spring_realdyn_realnet.py
```



### Network Completetion

The basic idea of the network completion algorithm is to set the initial states of the unobserved nodes as a set of new learnable parameters. Therefore, we can use the similar method with AIDD to learn the missing partial network and the initial states of the unobserved nodes. The parameters of dynamics learner can be also fine tuned.

#### Data Generation

The files whose name starts with "generate" are files that generate data.For example, if you want to generate data of voter, you can run the file generate_voter.py

```python
python generate_voter.py
or
nohup python -u generate_voter.py > generate_voter.txt 2>&1 & #Save the output file to a text file
```

#### Run Experiment

The files whose name starts with "train" are files that train. For example, if you want to run the experiment of voter, you can run the file train_voter_completetion.py

```python
python train_voter_completetion.py
or
nohup python -u train_voter_completetion.py > train_voter_completetion.txt 2>&1 & #Save the output file to a text file
```



## Cite

If you use this code in your own work, please cite our paper
```
@article{zhang2021automated,
  title={Automated Discovery of Interactions and Dynamics for Large Networked Dynamical Systems},
  author={Zhang, Yan and Guo, Yu and Zhang, Zhang and Chen, Mengyuan and Wang, Shuo and Zhang, Jiang},
  journal={arXiv preprint arXiv:2101.00179},
  year={2021}
}
```


