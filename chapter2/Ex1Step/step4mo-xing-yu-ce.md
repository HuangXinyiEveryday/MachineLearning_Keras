# 一、数据预测 —model.predict\(\)

* Keras提供**predict\(\)**方法对未知数据利用训练模型进行预测，输出层使用 Sigmoid 激活函数, 因此我们的预测值将会在 0 到 1 的区间内。在这个分类任务中,我们可以轻易地通过四舍五入转换为离散二分类。

```py
#预测模型，参数我们依旧采用训练集的参数
predictions=model.predict_classes(X)
score=model.evaluate(X,Y)
#打印准确率
print(score)
#预测结果与真实结果比较
print('预测结果  真实结果')
#按照标准格式打印
i=0
for p in predictions:
    print(' ',p,'   ',[Y[i]])
    i+=1
```

# 二、模型绘图

* Python的**matplotlib.pyplot**库可以绘制模型的acc和loss曲线

```py
#epochs表示我们只训练100轮，batch_size表示每次批处理20个数据
#这个地方加入callbacks，实现绘图,history为构建的绘图实例（代码方法过长请看具体代码）
model.fit(X,Y,epochs=100,batch_size=20,validation_data=(X,Y),callbacks=[history])
#绘制acc-loss曲线
history.loss_plot('epoch')
```

![](/assets/import3.png)

