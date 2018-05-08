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





