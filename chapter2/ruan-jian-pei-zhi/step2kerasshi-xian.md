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

二、构建模型

编译模型

训练模型

评估模型

