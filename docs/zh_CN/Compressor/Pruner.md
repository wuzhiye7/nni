# NNI 支持的剪枝算法

NNI 提供了一些支持细粒度权重剪枝和结构化的滤波器剪枝算法。 **细粒度的剪枝**通常会导致非结构化的模型，这需要特定的硬件或软件来加速这样的稀疏网络。 **滤波器剪枝**通过删除整个滤波器来实现加速。  NNI 还提供了算法来进行**剪枝规划**。


**细粒度剪枝**
* [Level Pruner](#level-pruner)

**滤波器剪枝**
* [Slim Pruner](#slim-pruner)
* [FPGM Pruner](#fpgm-pruner)
* [L1Filter Pruner](#l1filter-pruner)
* [L2Filter Pruner](#l2filter-pruner)
* [Activation APoZ Rank Filter Pruner](#activationAPoZRankFilter-pruner)
* [Activation Mean Rank Filter Pruner](#activationmeanrankfilter-pruner)
* [Taylor FO On Weight Pruner](#taylorfoweightfilter-pruner)

**剪枝计划**
* [AGP Pruner](#agp-pruner)
* [NetAdapt Pruner](#netadapt-pruner)
* [SimulatedAnnealing Pruner](#simulatedannealing-pruner)
* [AutoCompress Pruner](#autocompress-pruner)

**其它**
* [ADMM Pruner](#admm-pruner)
* [Lottery Ticket 假设](#lottery-ticket-hypothesis)

## Level Pruner

This is one basic one-shot pruner: you can set a target sparsity level (expressed as a fraction, 0.6 means we will prune 60% of the weight parameters).

首先按照绝对值对指定层的权重排序。 然后按照所需的稀疏度，将值最小的权重屏蔽为 0。

### 用法

TensorFlow 代码
```python
from nni.compression.tensorflow import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model_graph, config_list)
pruner.compress()
```

PyTorch 代码
```python
from nni.compression.torch import LevelPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['default'] }]
pruner = LevelPruner(model, config_list)
pruner.compress()
```

#### Level Pruner 的用户配置

##### PyTorch

```eval_rst
..  autoclass:: nni.compression.torch.LevelPruner
```

##### Tensorflow

```eval_rst
..  autoclass:: nni.compression.tensorflow.LevelPruner
```


## Slim Pruner

这是一次性的 Pruner，在 ['Learning Efficient Convolutional Networks through Network Slimming'](https://arxiv.org/pdf/1708.06519.pdf) 中提出，作者 Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan 以及 Changshui Zhang。

![](../../img/slim_pruner.png)

> Slim Pruner **会遮盖卷据层通道之后 BN 层对应的缩放因子**，训练时在缩放因子上的 L1 正规化应在批量正规化 (BN) 层之后来做。BN 层的缩放因子在修剪时，是**全局排序的**，因此稀疏模型能自动找到给定的稀疏度。

### 用法

PyTorch 代码

```python
from nni.compression.torch import SlimPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['BatchNorm2d'] }]
pruner = SlimPruner(model, config_list)
pruner.compress()
```

#### Slim Pruner 的用户配置

##### PyTorch

```eval_rst
..  autoclass:: nni.compression.torch.SlimPruner
```

### 重现实验

我们实现了 ['Learning Efficient Convolutional Networks through Network Slimming'](https://arxiv.org/pdf/1708.06519.pdf) 中的一项实验。根据论文，对 CIFAR-10 上的 **VGGNet** 剪除了 $70\%$ 的通道，即约 $88.5\%$ 的参数。 实验结果如下：

| 模型            | 错误率(论文/我们的) | 参数量    | 剪除率   |
| ------------- | ----------- | ------ | ----- |
| VGGNet        | 6.34/6.40   | 20.04M |       |
| Pruned-VGGNet | 6.20/6.26   | 2.03M  | 88.5% |

实验代码在 [examples/model_compress](https://github.com/microsoft/nni/tree/master/examples/model_compress/)

***

## FPGM Pruner

这是一种一次性的 Pruner，FPGM Pruner 是论文 [Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration](https://arxiv.org/pdf/1811.00250.pdf) 的实现

FPGMPruner prune filters with the smallest geometric median.

 ![](../../img/fpgm_fig1.png)
> 以前的方法使用 “smaller-norm-less-important” 准则来修剪卷积神经网络中规范值较小的。 本文中，分析了基于规范的准则，并指出其所依赖的两个条件不能总是满足：(1) 滤波器的规范偏差应该较大；(2) 滤波器的最小规范化值应该很小。 为了解决此问题，提出了新的滤波器修剪方法，即 Filter Pruning via Geometric Median (FPGM)，可不考虑这两个要求来压缩模型。 与以前的方法不同，FPGM 通过修剪冗余的，而不是相关性更小的部分来压缩 CNN 模型。

### 用法

TensorFlow 代码
```python
from nni.compression.tensorflow import FPGMPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2D']
}]
pruner = FPGMPruner(model, config_list)
pruner.compress()
```

PyTorch 代码
```python
from nni.compression.torch import FPGMPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = FPGMPruner(model, config_list)
pruner.compress()
```

#### FPGM Pruner 的用户配置

##### PyTorch
```eval_rst
..  autoclass:: nni.compression.torch.FPGMPruner
```

##### Tensorflow
```eval_rst
..  autoclass:: nni.compression.tensorflow.FPGMPruner
```

## L1Filter Pruner

这是一种一次性的 Pruner，由 ['PRUNING FILTERS FOR EFFICIENT CONVNETS'](https://arxiv.org/abs/1608.08710) 提出，作者 Hao Li, Asim Kadav, Igor Durdanovic, Hanan Samet 和 Hans Peter Graf。

![](../../img/l1filter_pruner.png)

> L1Filter Pruner 修剪**卷积层**中的滤波器
> 
> 从第 i 个卷积层修剪 m 个滤波器的过程如下：
> 
> 1. 对于每个滤波器 ![](http://latex.codecogs.com/gif.latex?F_{i,j})，计算其绝对内核权重之和![](http://latex.codecogs.com/gif.latex?s_j=\sum_{l=1}^{n_i}\sum|K_l|)
> 2. 将滤波器按 ![](http://latex.codecogs.com/gif.latex?s_j) 排序。
> 3. 修剪 ![](http://latex.codecogs.com/gif.latex?m) 具有最小求和值及其相应特征图的滤波器。 在 下一个卷积层中，被剪除的特征图所对应的内核也被移除。
> 4. 为第 ![](http://latex.codecogs.com/gif.latex?i) 和 ![](http://latex.codecogs.com/gif.latex?i+1) 层创建新的内核举证，并保留剩余的内核 权重，并复制到新模型中。

### 用法

PyTorch 代码

```python
from nni.compression.torch import L1FilterPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
pruner = L1FilterPruner(model, config_list)
pruner.compress()
```

#### L1Filter Pruner 的用户配置

##### PyTorch
```eval_rst
..  autoclass:: nni.compression.torch.L1FilterPruner
```

### 重现实验

我们通过 **L1FilterPruner** 实现了 ['PRUNING FILTERS FOR EFFICIENT CONVNETS'](https://arxiv.org/abs/1608.08710) 中的一项实验， 即论文中，在 CIFAR-10 数据集上修剪 **VGG-16** 的 **VGG-16-pruned-A**，其中大约剪除了 $64\%$ 的参数。 实验结果如下：

| 模型              | 错误率(论文/我们的) | 参数量      | 剪除率   |
| --------------- | ----------- | -------- | ----- |
| VGG-16          | 6.75/6.49   | 1.5x10^7 |       |
| VGG-16-pruned-A | 6.60/6.47   | 5.4x10^6 | 64.0% |

实验代码在 [examples/model_compress](https://github.com/microsoft/nni/tree/master/examples/model_compress/)

***

## L2Filter Pruner

这是一种结构化剪枝算法，用于修剪权重的最小 L2 规范滤波器。 它被实现为一次性修剪器。

### 用法

PyTorch 代码

```python
from nni.compression.torch import L2FilterPruner
config_list = [{ 'sparsity': 0.8, 'op_types': ['Conv2d'] }]
pruner = L2FilterPruner(model, config_list)
pruner.compress()
```

### L2Filter Pruner 的用户配置

##### PyTorch
```eval_rst
..  autoclass:: nni.compression.torch.L2FilterPruner
```
***

## ActivationAPoZRankFilter Pruner

ActivationAPoZRankFilter Pruner is a pruner which prunes the filters with the smallest importance criterion `APoZ` calculated from the output activations of convolution layers to achieve a preset level of network sparsity. 剪枝标准 `APoZ` 的解释在论文 [Network Trimming: A Data-Driven Neuron Pruning Approach towards Efficient Deep Architectures](https://arxiv.org/abs/1607.03250) 中。

APoZ 定义为：

![](../../img/apoz.png)

### 用法

PyTorch 代码

```python
from nni.compression.torch import ActivationAPoZRankFilterPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = ActivationAPoZRankFilterPruner(model, config_list, statistics_batch_num=1)
pruner.compress()
```

注意：ActivationAPoZRankFilterPruner 用于修剪深度神经网络中的卷积层，因此 `op_types` 字段仅支持卷积层。

You can view [example](https://github.com/microsoft/nni/blob/master/examples/model_compress/model_prune_torch.py) for more information.

### User configuration for ActivationAPoZRankFilter Pruner

##### PyTorch
```eval_rst
..  autoclass:: nni.compression.torch.ActivationAPoZRankFilterPruner
```
***


## ActivationMeanRankFilter Pruner

ActivationMeanRankFilterPruner 是从卷积层激活的输出，用最小的重要性标准`平均激活`来修剪滤波器，来达到预设的网络稀疏度。 剪枝标准`平均激活`，在论文 [Pruning Convolutional Neural Networks for Resource Efficient Inference](https://arxiv.org/abs/1611.06440) 的 2.2 节中进行了介绍。 本文中提到的其他修剪标准将在以后的版本中支持。

### 用法

PyTorch 代码

```python
from nni.compression.torch import ActivationMeanRankFilterPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = ActivationMeanRankFilterPruner(model, config_list, statistics_batch_num=1)
pruner.compress()
```

注意：ActivationMeanRankFilterPruner 用于修剪深度神经网络中的卷积层，因此 `op_types` 字段仅支持卷积层。

You can view [example](https://github.com/microsoft/nni/blob/master/examples/model_compress/model_prune_torch.py) for more information.

### ActivationMeanRankFilterPruner 的用户配置

##### PyTorch
```eval_rst
..  autoclass:: nni.compression.torch.ActivationMeanRankFilterPruner
```
***

## TaylorFOWeightFilter Pruner

TaylorFOWeightFilter Pruner is a pruner which prunes convolutional layers based on estimated importance calculated from the first order taylor expansion on weights to achieve a preset level of network sparsity. 滤波器的估计重要性在论文 [Importance Estimation for Neural Network Pruning](http://jankautz.com/publications/Importance4NNPruning_CVPR19.pdf) 中有定义。 本文中提到的其他修剪标准将在以后的版本中支持。
>

![](../../img/importance_estimation_sum.png)

### 用法

PyTorch 代码

```python
from nni.compression.torch import TaylorFOWeightFilterPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = TaylorFOWeightFilterPruner(model, config_list, statistics_batch_num=1)
pruner.compress()
```


#### User configuration for TaylorFOWeightFilter Pruner

##### PyTorch
```eval_rst
..  autoclass:: nni.compression.torch.TaylorFOWeightFilterPruner
```
***


## AGP Pruner

This is an iterative pruner, In [To prune, or not to prune: exploring the efficacy of pruning for model compression](https://arxiv.org/abs/1710.01878), authors Michael Zhu and Suyog Gupta provide an algorithm to prune the weight gradually.
> We introduce a new automated gradual pruning algorithm in which the sparsity is increased from an initial sparsity value si (usually 0) to a final sparsity value sf over a span of n pruning steps, starting at training step t0 and with pruning frequency ∆t: ![](../../img/agp_pruner.png)
> The binary weight masks are updated every ∆t steps as the network is trained to gradually increase the sparsity of the network while allowing the network training steps to recover from any pruning-induced loss in accuracy. In our experience, varying the pruning frequency ∆t between 100 and 1000 training steps had a negligible impact on the final model quality. Once the model achieves the target sparsity sf , the weight masks are no longer updated. The intuition behind this sparsity function in equation (1).

### Usage

You can prune all weight from 0% to 80% sparsity in 10 epoch with the code below.

PyTorch code
```python
from nni.compression.torch import AGPPruner
config_list = [{
    'initial_sparsity': 0,
    'final_sparsity': 0.8,
    'start_epoch': 0,
    'end_epoch': 10,
    'frequency': 1,
    'op_types': ['default']
}]

# load a pretrained model or train a model before using a pruner
# model = MyModel()
# model.load_state_dict(torch.load('mycheckpoint.pth'))

# AGP pruner prunes model while fine tuning the model by adding a hook on
# optimizer.step(), so an optimizer is required to prune the model.
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

pruner = AGPPruner(model, config_list, optimizer, pruning_algorithm='level')
pruner.compress()
```

AGP pruner uses `LevelPruner` algorithms to prune the weight by default, however you can set `pruning_algorithm` parameter to other values to use other pruning algorithms:
* `level`: LevelPruner
* `slim`: SlimPruner
* `l1`: L1FilterPruner
* `l2`: L2FilterPruner
* `fpgm`: FPGMPruner
* `taylorfo`: TaylorFOWeightFilterPruner
* `apoz`: ActivationAPoZRankFilterPruner
* `mean_activation`: ActivationMeanRankFilterPruner

You should add code below to update epoch number when you finish one epoch in your training code.

PyTorch code
```python
pruner.update_epoch(epoch)
```
You can view [example](https://github.com/microsoft/nni/blob/master/examples/model_compress/model_prune_torch.py) for more information.

#### User configuration for AGP Pruner

##### PyTorch

```eval_rst
..  autoclass:: nni.compression.torch.AGPPruner
```

##### Tensorflow

```eval_rst
..  autoclass:: nni.compression.tensorflow.AGPPruner
```

***

## NetAdapt Pruner
NetAdapt allows a user to automatically simplify a pretrained network to meet the resource budget. Given the overall sparsity, NetAdapt will automatically generate the sparsities distribution among different layers by iterative pruning.

For more details, please refer to [NetAdapt: Platform-Aware Neural Network Adaptation for Mobile Applications](https://arxiv.org/abs/1804.03230).

![](../../img/algo_NetAdapt.png)

#### Usage

PyTorch code

```python
from nni.compression.torch import NetAdaptPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = NetAdaptPruner(model, config_list, short_term_fine_tuner=short_term_fine_tuner, evaluator=evaluator,base_algo='l1', experiment_data_dir='./')
pruner.compress()
```

You can view [example](https://github.com/microsoft/nni/blob/master/examples/model_compress/auto_pruners_torch.py) for more information.

#### User configuration for NetAdapt Pruner

##### PyTorch

```eval_rst
..  autoclass:: nni.compression.torch.NetAdaptPruner
```


## SimulatedAnnealing Pruner

We implement a guided heuristic search method, Simulated Annealing (SA) algorithm, with enhancement on guided search based on prior experience. The enhanced SA technique is based on the observation that a DNN layer with more number of weights often has a higher degree of model compression with less impact on overall accuracy.

- Randomly initialize a pruning rate distribution (sparsities).
- While current_temperature < stop_temperature:
    1. generate a perturbation to current distribution
    2. Perform fast evaluation on the perturbated distribution
    3. accept the perturbation according to the performance and probability, if not accepted, return to step 1
    4. cool down, current_temperature <- current_temperature * cool_down_rate

For more details, please refer to [AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates](https://arxiv.org/abs/1907.03141).

#### Usage

PyTorch code

```python
from nni.compression.torch import SimulatedAnnealingPruner
config_list = [{
    'sparsity': 0.5,
    'op_types': ['Conv2d']
}]
pruner = SimulatedAnnealingPruner(model, config_list, evaluator=evaluator, base_algo='l1', cool_down_rate=0.9, experiment_data_dir='./')
pruner.compress()
```

You can view [example](https://github.com/microsoft/nni/blob/master/examples/model_compress/auto_pruners_torch.py) for more information.

#### User configuration for SimulatedAnnealing Pruner

##### PyTorch

```eval_rst
..  autoclass:: nni.compression.torch.SimulatedAnnealingPruner
```


## AutoCompress Pruner
For each round, AutoCompressPruner prune the model for the same sparsity to achive the overall sparsity:

        1. Generate sparsities distribution using SimulatedAnnealingPruner
        2. Perform ADMM-based structured pruning to generate pruning result for the next round.
           Here we use `speedup` to perform real pruning.

For more details, please refer to [AutoCompress: An Automatic DNN Structured Pruning Framework for Ultra-High Compression Rates](https://arxiv.org/abs/1907.03141).

#### Usage

PyTorch code

```python
from nni.compression.torch import ADMMPruner
config_list = [{
        'sparsity': 0.5,
        'op_types': ['Conv2d']
    }]
pruner = AutoCompressPruner(
            model, config_list, trainer=trainer, evaluator=evaluator,
            dummy_input=dummy_input, num_iterations=3, optimize_mode='maximize', base_algo='l1',
            cool_down_rate=0.9, admm_num_iterations=30, admm_training_epochs=5, experiment_data_dir='./')
pruner.compress()
```

You can view [example](https://github.com/microsoft/nni/blob/master/examples/model_compress/auto_pruners_torch.py) for more information.

#### User configuration for AutoCompress Pruner

##### PyTorch

```eval_rst
..  autoclass:: nni.compression.torch.AutoCompressPruner
```


## ADMM Pruner
Alternating Direction Method of Multipliers (ADMM) is a mathematical optimization technique, by decomposing the original nonconvex problem into two subproblems that can be solved iteratively. In weight pruning problem, these two subproblems are solved via 1) gradient descent algorithm and 2) Euclidean projection respectively.

During the process of solving these two subproblems, the weights of the original model will be changed. An one-shot pruner will then be applied to prune the model according to the config list given.

This solution framework applies both to non-structured and different variations of structured pruning schemes.

For more details, please refer to [A Systematic DNN Weight Pruning Framework using Alternating Direction Method of Multipliers](https://arxiv.org/abs/1804.03294).

#### Usage

PyTorch code

```python
from nni.compression.torch import ADMMPruner
config_list = [{
            'sparsity': 0.8,
            'op_types': ['Conv2d'],
            'op_names': ['conv1']
        }, {
            'sparsity': 0.92,
            'op_types': ['Conv2d'],
            'op_names': ['conv2']
        }]
pruner = ADMMPruner(model, config_list, trainer=trainer, num_iterations=30, epochs=5)
pruner.compress()
```

You can view [example](https://github.com/microsoft/nni/blob/master/examples/model_compress/auto_pruners_torch.py) for more information.

#### User configuration for ADMM Pruner

##### PyTorch

```eval_rst
..  autoclass:: nni.compression.torch.ADMMPruner
```


## Lottery Ticket 假设
[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635), authors Jonathan Frankle and Michael Carbin,provides comprehensive measurement and analysis, and articulate the *lottery ticket hypothesis*: dense, randomly-initialized, feed-forward networks contain subnetworks (*winning tickets*) that -- when trained in isolation -- reach test accuracy comparable to the original network in a similar number of iterations.

In this paper, the authors use the following process to prune a model, called *iterative prunning*:
> 1. Randomly initialize a neural network f(x;theta_0) (where theta_0 follows D_{theta}).
> 2. Train the network for j iterations, arriving at parameters theta_j.
> 3. Prune p% of the parameters in theta_j, creating a mask m.
> 4. Reset the remaining parameters to their values in theta_0, creating the winning ticket f(x;m*theta_0).
> 5. Repeat step 2, 3, and 4.

If the configured final sparsity is P (e.g., 0.8) and there are n times iterative pruning, each iterative pruning prunes 1-(1-P)^(1/n) of the weights that survive the previous round.

### 用法

PyTorch code
```python
from nni.compression.torch import LotteryTicketPruner
config_list = [{
    'prune_iterations': 5,
    'sparsity': 0.8,
    'op_types': ['default']
}]
pruner = LotteryTicketPruner(model, config_list, optimizer)
pruner.compress()
for _ in pruner.get_prune_iterations():
    pruner.prune_iteration_start()
    for epoch in range(epoch_num):
        ...
```

The above configuration means that there are 5 times of iterative pruning. As the 5 times iterative pruning are executed in the same run, LotteryTicketPruner needs `model` and `optimizer` (**Note that should add `lr_scheduler` if used**) to reset their states every time a new prune iteration starts. Please use `get_prune_iterations` to get the pruning iterations, and invoke `prune_iteration_start` at the beginning of each iteration. `epoch_num` is better to be large enough for model convergence, because the hypothesis is that the performance (accuracy) got in latter rounds with high sparsity could be comparable with that got in the first round.


*Tensorflow version will be supported later.*

#### User configuration for LotteryTicket Pruner

##### PyTorch

```eval_rst
..  autoclass:: nni.compression.torch.LotteryTicketPruner
```

### Reproduced Experiment

We try to reproduce the experiment result of the fully connected network on MNIST using the same configuration as in the paper. The code can be referred [here](https://github.com/microsoft/nni/tree/master/examples/model_compress/lottery_torch_mnist_fc.py). In this experiment, we prune 10 times, for each pruning we train the pruned model for 50 epochs.

![](../../img/lottery_ticket_mnist_fc.png)

The above figure shows the result of the fully connected network. `round0-sparsity-0.0` is the performance without pruning. Consistent with the paper, pruning around 80% also obtain similar performance compared to non-pruning, and converges a little faster. If pruning too much, e.g., larger than 94%, the accuracy becomes lower and convergence becomes a little slower. A little different from the paper, the trend of the data in the paper is relatively more clear.
