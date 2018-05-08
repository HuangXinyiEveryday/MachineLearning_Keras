一、加载数据

* 使用panda库的read\_csv读取本地csv文件（需要加载panda和numpy基础库、keras模块）

```py
#加载数据
dataset=pd.read_csv("diabetes.csv")
#将数据分成输入和输出两组，进行交叉验证。
#前8个特征值为输入X，第九个特征值outcome为输出Y
X=dataset.iloc[:,0:8]
Y=dataset.iloc[:,8]
```

# 二、构建模型

* Keras中的模型被定义为一系列的层，我们需要实例化一个Sequential模型对象，每次添加一层直到对网络的拓扑结构满意。

**关键在于确定的是输入层的数目，层数量和类型如何确定？**

* 这是启发式的过程,我们需要通过不断地试错找出最好的网络结构、一般来说,需要足够大的网络去明白结构对于问题是否有用。

* 本实例使用三层全连接的结构，Keras使用Dense定义全连接层第一个参数定义层的神经元数量,第二个参数 init 定义权重的初始化方法, activation 参数定义激活函数。

```py
#创建模型
model=Sequential()#实例化Sequential 模型对象
#input_dim设置输入层数量，设置为8代表8个输入变量;第一个参数为本层神经元的数量
#init=uniform表示权重初始化成一个服从均匀分布的小随机数。Keras 标准均匀分布权重初始值[0,0.5];
#init=normal则表示从高斯分布（正态分布）中产生一个小的随机数进行权重初始化。
#activation='relu'使用线性整流函数relu，sigmoid是S型函数作为激活函数
model.add(Dense(12,input_dim=8,kernel_initializer='uniform',activation='relu'))
model.add(Dense(8,kernel_initializer='uniform',activation='relu'))
model.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
```

![](/assets/import1.png)

# 三、编译模型

* 定义好的模型可以编译：Keras会调用Theano或者TensorFlow编译模型。后端会自动选择表示网络的最佳方法，配合你的硬件，在这里使用TensorFlow backend。
* 注：损失函数和优化算法是后台TensorFlow提供的，因此想要模型足够精确，需要深入了解使用TensorFlow或者自己开发API后台接口；**训练神经网络的意义是找到最好的一组权重**，解决问题

```py
#编译模型
#定义损失函数和优化算法以及需要收集的数据，‘binary_crossentropy’错误的对数作为损失函数，adam作为优化算法
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
```

# 四、训练模型

* 调用Keras模型的提供的**fit\(\)方法**即可开始训练。 网络按轮训练，通过**epochs**参数控制。每次送入的数据（一次处理的数据量）可以用batch\_size参数控制。

```py
#训练模型
#nb_epoch表示我们只训练150轮，batch_size表示每次批处理10个数据
model.fit(X,Y,epochs=150,batch_size=10)
```

# 五、评估模型

* 测试数据拿出来检验一下模型的效果，但不能衡量模型的预测能力。但需要注意不能将所有测试数据训练后在进行直接测试。应该将数据分成训练和测试集

* 采用Keras自动验证，Keras可以将数据自动分出一部分，每次训练后进行验证。在训练时用validation\_split参数可以指定验证数据的比例，一般是总数据的20%或者33%。

* 调用Keras模型的**evaluate\(\)方法**，传入训练时的数据。输出是平均值，包括平均误差和其他的数据，例如准确度等，可以自己选择评估结果参数打印输出

```py
#调整训练数据集
model.fit(X,Y,validation_split=0.33,epochs=150,batch_size=10)
#评估模型
scores=model.evaluate(X,Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))#得到准确度
```



