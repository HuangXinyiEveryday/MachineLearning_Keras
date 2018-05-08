# 一、交叉验证模型—Keras+Scikit-Learn

* Keras本身提供的evaluate\(\)评估方法只是通过在有限的测试集数据上模型的准确度，不能很好度量我们模型的预测性能，因此借助_Scikit-Learn_的库进行交叉验证的方式验证模型的准确度

* Keras的**KerasClassifier**和**KerasRegressor**两个类接受build\_fn参数，传入编译好的模型。加入epochs=150和batch\_size=10，这两个参数会传入模型的fit\(\)方法。

* 用Scikit-learn的**StratifiedKFold**类进行10折交叉验证，**测试模型在未知数据的性能**，并使用cross\_val\_score\(\)函数检测模型，打印结果。

```py
#交叉验证模型
model=KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=20)
#n_folds可以定义进行几折交叉验证;shuffle默认False,会对数据产生随机搅动(洗牌)
kfold=StratifiedKFold(n_splits=10,shuffle=False,random_state=seed)
#第一个参数是分类器;cv表示不同的交叉验证方法；
results=cross_val_score(model,X,Y,cv=kfold)
#结果即我们的分类模型得到交叉验证10次的平均准确率
print(results.mean())
```

# 二、优化模型

### 1.网格搜索

* 我们可以给fit\(\)方法传入参数，KerasClassifier的build\_fn方法也可以传入参数。可以利用这点进一步调整模型。使用网格搜索测试不同参数的性能，使用不同的优化算法和初始权重调整网络

### 2.优化模型步骤

* 优化算法：搜索权重的方法
* 初始权重：初始化不同的网络
* 训练次数：对模型训练的次数
* 批处理大小：每一轮中每次批处理多少数据

# 3.Scikit-Learn优化模型

* 所有的参数组成一个字典，传入scikit-learn的**GridSearchCV类**：GridSearchCV会对每组参数（2×3×3×3）进行训练，进行3折交叉检验。

```py
#测试不同参数，优化模型
model=KerasClassifier(build_fn=create_model, nb_epoch=150, batch_size=10)
optimizers = ['rmsprop', 'adam']#优化算法[2]
init = ['glorot_uniform', 'normal', 'uniform']#权重初始化算法[3]
epochs = np.array([50, 100, 150])#训练轮数[3]
batches = np.array([5, 10, 20])#一次批处理几条数据[3]
param_grid = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches, init=init) 
#创建GridSearchCV，用来进行不同参数数据分析
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, Y)
```

### 4.网格搜索缺点

* 网格搜索的缺点：计算量巨大、耗时巨长。如果模型小还可以取一部分数据试试。比如本实例\(网络和数据集都不大:1000个数据内，9个参数\)。最后scikit-learn会输出最好的参数和模型，以及平均值。

### 三、自定义API优化参数

如果不使用SL提供的网格搜索进行优化，针对三层全感知网络我们可以自己优化的参数如下

###### 1.模型建立阶段

* optimizers-优化算法（可调用TensorFlow接口，也可以自己定义优化算法）
* init—权重初始化算法
* activation—激活函数

1. epochs—fit训练轮数
2. batches—每轮每次批处理数据量
3. 


