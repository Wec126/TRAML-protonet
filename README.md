# TRAML-protonet

本代码库根据CVPR 2020《Boosting Few-Shot Learning With Adaptive Margin Loss》论文中对TRAML的定义在Prototypical-Network初步实现其功能，但是由于源代码未开源，因此在本代码在复现结果上无法到达元论文中的效果。

## 1. 论文介绍

《Boosting Few-Shot Learning With Adaptive Margin Loss》中主要介绍了TRAML，可以利用自适应边际损失增强小样本学习

### 1.1 背景介绍

由于在现实世界中，很多图像识别的场景中没有大量的数据可供模型进行学习，收到人类利用小样本即可识别新类能力的启发，人们开始研究小样本学习问题(FSL: Few Shot Learning)，即在没有大量训练数据的情况下如何让深度神经网络也能像人类一样，把过往的经验迁移到新的类别中。

其中，基于度量学习的元学习方法在小样本学习上表现了很好的性能，它通过学习一个好的特征表示，使得在特征空间中，同类样本聚集，异类样本分开。这样，只需要简单通过和各类样本的距离比较，就能预测一个新类样本所属的类别。

但是在已有的方法中，相似类别的样本常常在特征空间里的距离挨得很近，大大限制了分类精度。本文提出在类别之间加入自适应的边际距离来提升基于度量学习的元学习边际距离是通过类别之间的语义相似度自动生成的。直观上，语义上越相似的类别之间越难区分，设定的边际距离也应该越大。

![image](https://github.com/Wec126/TRAML-protonet/assets/57513224/6c1e6562-9ac7-49a4-baa5-98682f1953f6)


[上图为使用自适应边际示意图]

### 1.2 方法介绍

元学习(meta-learning)是一种处理小样本学习的常用框架，它包含meta-training和meta-testing 两个阶段。在meta-learning阶段，模型按照一个个episode来训练：在每个episode中，首先构造一个task，该task从整个base class数据集中抽取一些样本来组成一个小训练集和小测试集，然后用它来更新模型。

在meta-tesing阶段，我们用到学到的模型来预测novel class中的样本。

近年来，基于度量学习的元学习方法变得很流行，它假设存在这样一个embedding space，每个类别的样本聚集在一个代表点(class representation)周围，而这些类代表点当作每个类的参考点来预测测试样本的标签。

#### 损失函数

在meta-training阶段，我们从base class数据集中随机抽取$` n_t `$个类，类别集合记作$` C_t `$，每个类中随机抽取$`n_s`$个样本，合并后作为小训练集，成为支持集$`S`$。然后再从每个类随机抽取若干样本，合并后作为小测试集，成为问询集$`Q`$。支持集和问询集中的样本通过embedding函数$F$映射到特征空间中，每个类的代表点用$`r_1,...,r_{n_t}`$来表示。最后引入一个度量模板$`D`$来衡量两个特征向量相似度(本代码中使用cosine similarity)

对于每个问询集中的点$`(x,y)∈Q`$，我们计算他的特征向量到各个类代表点之间到相似度，然后通过softmax计算损失函数如下：

$$ L^{cls}=-\frac {1}{|Q|} \sum_{(x,y)∈Q}log \frac{e^{D(F(x),r_y)}}{\sum_{k∈C_t}e^{D(F(x),r_k)}} $$

### 1.3 自适应边际损失(Adaptive Margin Loss)

为了更好地分开各个类别，一个最简单的加margin的方法是：

$$ L^{na}=-\frac{1}{|Q|}\sum_{(x,y)∈Q}log\frac{e^{D(F(x),r_y)}}{e^{D(F(x,r_y))+\sum_{k∈C_t \diagdown \{y\}}e^{D(F(x),r_k)+m}}} $$

上述loss成为Native Additive Margin Loss(NAML)，它在类别两两之间加上了相同的边际$m$，强迫不同类的样本之间分开一定的距离。但是这种简单加上等距离边际的方法在小样本可能会带来错误。

为了进一步精细化设计边际，作者借助类别之间的语义相似度，来自适应地生成边际

### 1.4 类别相关的边际损失(CRAML)

对于两个类别$i$和$j$，首先得到他们的语义向量$`e_i`$和$`e_j`$，然后我们通过线性模型$M$来生成他们的边际，即$`m^{cr}_{i,j}:=M(e_i,e_j)=\alpha \centerdot sim(e_i,e_j)+\beta`$其中$\alpha$和$`\beta`$是要学习的参数，于是我们将损失函数改写成 $$L^{na}=-\frac{1}{|Q|}\sum_{(x,y)∈Q}log\frac{e^{D(F(x),r_y)}}{e^{D(F(x,r_y))+\sum_{k∈C_t \diagdown \{y\}}e^{D(F(x),r_k)+m^{cr}_{y,k}}}}$$

通过合适地引入语义信息，CRAML可以把相似地类别在特征空间中分的更开，从而帮助更好地标识新类的样本

![image](https://github.com/Wec126/TRAML-protonet/assets/57513224/9282f25f-e0ea-48d4-a6d1-ae14e6c2231a)


### 1.5 任务相关的边际损失(TRAML)

到目前为止，我们都只考虑边际与任务无关。如果每次只考虑一个meta-training task中涉及到的类别，那么可以更加精细地生成合适地边际。通过将一个meta-training task中的每个类与该meta-training task中其他类一一对比，我们可以衡量一个task内“相对的”语义相似度，从而生成适合这个task的边际。

![image](https://github.com/Wec126/TRAML-protonet/assets/57513224/bc4cfe82-ae32-4903-b6b2-71833633abcb)


具体来说，对于一个meta-training task中的类$`y∈C_t`$，我们用一个神经网络$G$来生成task内的边际，即$`\{m^{tr}_{y,k}\}_{y∈C_t\diagdown \{y\}}=G(\{sim(e_y,e_k)\}_{k∈C_t\diagdown \{y\}})`$

损失函数对应地改写为：

$L^{tr}= -\frac{1}{|Q|}\sum_{(x,y)∈Q}log \frac{e^{D(F(x),r_y)}}{e^{D(F(x),r_y)+\sum_{k∈C_t\diagdown \{y\}}e^{D(F(x)),r_k}+m^{tr}_{y,k}}}$

也就是说，对于同一个问询集样本$(x,y)∈Q$，我们首先计算它和task内其他每个类的语义相似度，然后把这些语义相似度通过神经网络$G$来生成损失函数需要的边际，最后累加到损失函数TRAML中

(由于神经网络$G$的细节没有开源，因此在具体实验中我们是用最简单的线性模型来进行测试)

## 2. Pytorch版本

本代码基于Omniglot数据集进行训练和测试，以下为执行过程

```
python train.py
```

| Avg Train Loss      | Avg Train Acc     | Avg Val Loss        | Avg Val Acc        |
| ------------------- | ----------------- | ------------------- | ------------------ |
| 0.17820393593583314 | 0.601433334350586 | 0.18183235862867758 | 0.8319999992847442 |

## 3. Mindspore版
Mindspore版本基于
```
https://gitee.com/mindspore/models/blob/r1.9/research/cv/ProtoNet
```
以上链接的代码改进<br>
<br>
为匹配用户习惯，MSAdapter设计目的是在用户不感知的情况下，能适配PyTorch代码运行在昇腾（Ascend）设备上。MSAdapter以PyTorch的接口为标准，为用户提供一套和PyTorch一样（接口和功能完全一致）的中高阶模型构建接口和数据处理接口。图2展示了MSAdapter的层次结构。其特点如下：<br>
轻量化封装<br>
接口和功能与PyTorch完全映射<br>
PyTorch代码少量修改可在Ascend运行<br>
支持MindSpore原生接口<br>
高效性能<br>
![](https://img-blog.csdnimg.cn/img_convert/1438a0c8b9499ed17eec278d67129f05.png)<br>
<br>
使用改进算法需要将<br>
```python
import torch
```
改为<br>
```python
import msadapter.pytorch as torch
```
最后直接运行<br>
```python
python train.python
```

