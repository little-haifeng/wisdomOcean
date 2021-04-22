# XGBoost参数

XGBoost作者将参数分为3类，运行XGBoost之 前需要设置这三类参数:

#### 1.一般 参数(General parameters): 

指导整体功能

#### 2.助推器参数(Booster parameters) :

在每个步骤指导个体助推器(树/回归)

#### 3.学习任务参数(Task parameters) :

指导执行任务的优化



## 一般参数

#### (1) booster [default= gbtree]，助推器[默认gbtree]

选择要在每次迭代时运行的模型类型。它有2个选项:
gbtree:基于树的模型，gblinear: 线性模型

#### (2) silent [default=0]，无声模式[default = 0]: 

静音模式激活设置为1，即不会打印正在运行的消息。
取0时表示打印出运行时信息，有助于理解模型。

#### (3) nthread [默认为未设置的最大线程数]

运行时的线程数(并行处理)。缺省值是当前系统可以获得的最大线程数

#### (4) num_ pbuffer:预测缓冲区大小，通常设置为训练实例的数目。缓冲用于保存最后一一步的预测结果，无需人为设置。

#### (5) num_ feature: Boosting过 程中用到的特征维数，设置为特征个数。XGBoost会 自动设置，无需人为设置。



## 助推器参数






- max depth [default -6]:数的最大深度，缺省值为6，取值范围为: [1,0]。 用于控
  制过拟合，因为更高的深度将允许模型学习特定于特定样本的关系，需要使用CV
  函数来进行调优，典型值: 3-10

- max leaf nodes:树中终端节点或叶子的最大数量，可以代替max depth。

- gamma [default -0] : Gamma指定节点分裂所需的最小损失函数下降值。这个参数
  的值越大，算法越保守，该值可能会根据损失函数而有所不同，因此应进行调整。
  取值 范围为: [0,∞]

- eta [default=0.3]:为了防止过拟合，更新过程中用到的收缩步长。在每次提升计算
  之后，算法会直接获得新特征的权重。eta通过缩减特征的权重使提升计算过程更
  加保守，使模型更健壮。缺省值为0.3，取值范围为: [0,1]

- min child. weight [default=1]:子节点最小样本权重和。如果一个叶子节点的样本
  权重和小于min child weight， 则拆分过程结束。在线性回归模型中，该参数是指
  建立每个模型所需要的最小样本数。可用于避免过拟合，值较大时可以避免模型学
  习到局部的特殊样本，但值过高时会导致欠拟合。可用CV来调整，范围: [0,∞]

-  subsample [default=1]:用于训练模型的子样本占整个样本集合的比例，能够防止.
  过拟合。，取值范围为: (0,1]

- colsample_ bytree [default=1]:在建立树时对特征采样的比例，缺省值为1，取值范
  围为: (0,1]

  **Linear Booster参数**

  - alpha [default =0]: L1 正则的惩罚系数
  - lambda [default =0]: L2正则的惩罚系数
  - lambda_ bias: 在偏置上的L2正则。缺省值为0


  



## 学习任务参数

- [ ] objective [ default reg:linear ]
  定义学习任务及相应的学习目标，可选的目标函数如下:
- [ ] reg:linear    线性回归。
- [ ] reg:logistic    逻辑回归。
- [ ] binary:logistic    二分 类的逻辑回归问题，输出为概率。
- [ ] binary:logitraw    二分 类的逻辑回归问题，输出的结果为wTx。
- [ ] count:poisson    计数问题的poisson回归，输出结果为poisson分布。
- [ ] multi:softmax    让XGBoost采用softmax目标函数处理多分类问题，同时需要设置参
  数num class (类别个数)
- [ ] multi:softprob    和softmax- 样， 输出的是ndata * nclass的向量，可以将该向量reshape成
  ndata行nclass列的矩阵。每行数据表示样本所属于每个类别的概率。
- [ ] rank:pairwiseset    XGBoost to do ranking task by minimizing the pairwise loss 

## 杂记

![image-20210422214301427](C:\Users\33309\AppData\Roaming\Typora\typora-user-images\image-20210422214301427.png)

![image-20210422214319340](C:\Users\33309\AppData\Roaming\Typora\typora-user-images\image-20210422214319340.png)

![image-20210422214326760](C:\Users\33309\AppData\Roaming\Typora\typora-user-images\image-20210422214326760.png)