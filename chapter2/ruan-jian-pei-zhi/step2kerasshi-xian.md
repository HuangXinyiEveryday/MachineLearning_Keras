# 一、加载数据

1.使用panda库的read\_csv读取本地csv文件（需要加载panda和numpy基础库、keras模块）

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

三、编译模型

# 训练模型

# 评估模型



