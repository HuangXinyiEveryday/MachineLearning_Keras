# 15个python进行机器学习的库

## 一、核心库

### 1）NumPy——地址：[http://www.numpy.org](http://www.numpy.org)

* 当使用 Python 开始处理科学任务时，不可避免地需要求助 Python 的 SciPy Stack，它是专门为 Python 中的科学计算而设计的软件的集合（不要与 SciPy 混淆，它只是这个 stack 的一部分，以及围绕这个 stack 的社区）。这个 stack 相当庞大，其中有十几个库，所以我们想聚焦在核心包上（特别是最重要的）。

* NumPy（代表 Numerical Python）是**构建科学计算 stack 的最基础的包**。它为 Python 中的 n 维数组和矩阵的操作提供了大量有用的功能。该库还提供了 NumPy 数组类型的数学运算向量化，可以提升性能，从而加快执行速度。

### 2）SciPy——地址：[https://www.scipy.org](https://www.scipy.org)

* SciPy 是一个**工程和科学软件库**。除此以外，你还要了解 SciPy Stack 和 SciPy 库之间的区别。SciPy 包含线性代数、优化、集成和统计的模块。SciPy 库的主要功能建立在 NumPy 的基础之上，因此它的数组大量使用了 NumPy。它通过其特定的子模块提供高效的数值例程操作，比如数值积分、优化和许多其他例程。SciPy 的所有子模块中的函数都有详细的文档，这也是一个优势。

### 3）Pandas——地址：[http://pandas.pydata.org](http://pandas.pydata.org)

* Pandas 是一个 Python 包，旨在通过「标记（labeled）」和「关系（relational）」数据进行工作，简单直观。Pandas 是 data wrangling 的完美工具。它设计用于**快速简单的数据操作、聚合和可视化**。库中有两个主要的数据结构：

Series：一维、Data Frames：二维

![](http://t12.baidu.com/it/u=2099500088,577537836&fm=170&s=A012709498B85588245A3CD9030010BC&w=345&h=131&img.JPEG)

例如，当你要从这两种类型的结构中接收到一个新的「Dataframe」类型的数据时，你将通过传递一个「Series」来将一行添加到「Dataframe」中来接收这样的 Dataframe

# 二、可视化

### 4）Matplotlib地址：[https://matplotlib.org](https://matplotlib.org)

* Matplotlib 是另一个 SciPy Stack 核心软件包和另一个 Python 库，专为**轻松生成简单而强大的可视化而量身定制**。它是一个顶尖的软件，使得 Python（在 NumPy、SciPy 和 Pandas 的帮助下）成为 MatLab 或 Mathematica 等科学工具的显著竞争对手。然而，这个库比较底层，这意味着你需要编写更多的代码才能达到高级的可视化效果，通常会比使用更高级工具付出更多努力，但总的来说值得一试。花一点力气，你就可以做到任何可视化：

* 线图散点图条形图和直方图饼状图茎图轮廓图场图频谱图，还有使用 Matplotlib 创建标签、网格、图例和许多其他格式化实体的功能。基本上，一切都是可定制的。该库支持不同的平台，并可使用不同的 GUI 工具套件来描述所得到的可视化。许多不同的 IDE（如 IPython）都支持 Matplotlib 的功能。

* 还有一些额外的库可以使可视化变得更加容易

### 5）Seaborn——地址：[https://seaborn.pydata.org](https://seaborn.pydata.org)

Seaborn 主要关注统计模型的可视化；这种可视化包括热度图（heat map），可以总结数据但也描绘总体分布。Seaborn 基于 Matplotlib，并高度依赖于它。

![](http://t11.baidu.com/it/u=282742669,3050531883&fm=170&s=4050C532451669CA1AD981CE0100C0B2&w=640&h=570&img.JPEG)

6）Bokeh

地址：[http://bokeh.pydata.org](http://bokeh.pydata.org)

Bokeh 也是一个很好的可视化库，其目的是交互式可视化。与之前的库相反，这个库独立于 Matplotlib。正如我们已经提到的那样，Bokeh 的重点是交互性，它通过现代浏览器以数据驱动文档（d3.js）的风格呈现。

![](http://t12.baidu.com/it/u=1047918197,246651124&fm=170&s=9741FE10000A53457E96F34C030070E5&w=433&h=431&img.JPEG)

7）Plotly

地址：[https://plot.ly](https://plot.ly)

最后谈谈 Plotly。它是一个基于 Web 的工具箱，用于构建可视化，将 API 呈现给某些编程语言（其中包括 Python）。在 plot.ly 网站上有一些强大的、开箱即用的图形。为了使用 Plotly，你需要设置你的 API 密钥。图形处理会放在服务器端，并在互联网上发布，但也有一种方法可以避免这么做。

![](http://t11.baidu.com/it/u=455752477,2300878776&fm=170&s=14F2EC3203124C67565DADD20000E0B3&w=640&h=345&img.JPEG)

机器学习

8）SciKit-Learn

地址：[http://scikit-learn.org](http://scikit-learn.org)

Scikits 是 SciPy Stack 的附加软件包，专为特定功能（如图像处理和辅助机器学习）而设计。在后者方面，其中最突出的一个是 scikit-learn。该软件包构建于 SciPy 之上，并大量使用其数学操作。

scikit-learn 有一个简洁和一致的接口，可利用常见的机器学习算法，让我们可以简单地在生产中应用机器学习。该库结合了质量很好的代码和良好的文档，易于使用且有着非常高的性能，是使用 Python 进行机器学习的实际上的行业标准。

深度学习：Keras / TensorFlow / Theano

在深度学习方面，Python 中最突出和最方便的库之一是 Keras，它可以在 TensorFlow 或者 Theano 之上运行。让我们来看一下它们的一些细节。

9）Theano

地址：[https://github.com/Theano](https://github.com/Theano)

首先，让我们谈谈 Theano。Theano 是一个 Python 包，它定义了与 NumPy 类似的多维数组，以及数学运算和表达式。该库是经过编译的，使其在所有架构上能够高效运行。这个库最初由蒙特利尔大学机器学习组开发，主要是为了满足机器学习的需求。

要注意的是，Theano 与 NumPy 在底层的操作上紧密集成。该库还优化了 GPU 和 CPU 的使用，使数据密集型计算的性能更快。

效率和稳定性调整允许更精确的结果，即使是非常小的值也可以，例如，即使 x 很小，log\(1+x\) 也能得到很好的结果。

10）TensorFlow

地址：[https://www.tensorflow.org](https://www.tensorflow.org)

TensorFlow 来自 Google 的开发人员，它是用于数据流图计算的开源库，专门为机器学习设计。它是为满足 Google 对训练神经网络的高要求而设计的，是基于神经网络的机器学习系统 DistBelief 的继任者。然而，TensorFlow 并不是谷歌的科学专用的——它也足以支持许多真实世界的应用。

TensorFlow 的关键特征是其多层节点系统，可以在大型数据集上快速训练人工神经网络。这为 Google 的语音识别和图像识别提供了支持。

11）Keras

地址：[https://keras.io](https://keras.io)

最后，我们来看看 Keras。它是一个使用高层接口构建神经网络的开源库，它是用 Python 编写的。它简单易懂，具有高级可扩展性。它使用 Theano 或 TensorFlow 作为后端，但 Microsoft 现在已将 CNTK（Microsoft 的认知工具包）集成为新的后端。

其简约的设计旨在通过建立紧凑型系统进行快速和容易的实验。

Keras 极其容易上手，而且可以进行快速的原型设计。它完全使用 Python 编写的，所以本质上很高层。它是高度模块化和可扩展的。尽管它简单易用且面向高层，但 Keras 也非常深度和强大，足以用于严肃的建模。

Keras 的一般思想是基于神经网络的层，然后围绕层构建一切。数据以张量的形式进行准备，第一层负责输入张量，最后一层用于输出。模型构建于两者之间。

自然语言处理

12）NLTK

地址：[http://www.nltk.org](http://www.nltk.org)

这套库的名称是 Natural Language Toolkit（自然语言工具包），顾名思义，它可用于符号和统计自然语言处理的常见任务。NLTK 旨在促进 NLP 及相关领域（语言学、认知科学和人工智能等）的教学和研究，目前正被重点关注。

NLTK 允许许多操作，例如文本标记、分类和 tokenizing、命名实体识别、建立语语料库树（揭示句子间和句子内的依存性）、词干提取、语义推理。所有的构建块都可以为不同的任务构建复杂的研究系统，例如情绪分析、自动摘要。

13）Gensim

地址：[http://radimrehurek.com/gensim](http://radimrehurek.com/gensim)

这是一个用于 Python 的开源库，实现了用于向量空间建模和主题建模的工具。这个库为大文本进行了有效的设计，而不仅仅可以处理内存中内容。其通过广泛使用 NumPy 数据结构和 SciPy 操作而实现了效率。它既高效又易于使用。

Gensim 的目标是可以应用原始的和非结构化的数字文本。Gensim 实现了诸如分层 Dirichlet 进程（HDP）、潜在语义分析（LSA）和潜在 Dirichlet 分配（LDA）等算法，还有 tf-idf、随机投影、word2vec 和 document2vec，以便于检查一组文档（通常称为语料库）中文本的重复模式。所有这些算法是无监督的——不需要任何参数，唯一的输入是语料库。

数据挖掘与统计

14）Scrapy

地址：[https://scrapy.org](https://scrapy.org)

Scrapy 是用于从网络检索结构化数据（如联系人信息或 URL）的爬虫程序（也称为 spider bots）的库。它是开源的，用 Python 编写。它最初是为 scraping 设计的，正如其名字所示的那样，但它现在已经发展成了一个完整的框架，可以从 API 收集数据，也可以用作通用的爬虫。

该库在接口设计上遵循著名的 Don』t Repeat Yourself 原则——提醒用户编写通用的可复用的代码，因此可以用来开发和扩展大型爬虫。

Scrapy 的架构围绕 Spider 类构建，该类包含了一套爬虫所遵循的指令。

15）Statsmodels

地址：[http://www.statsmodels.org](http://www.statsmodels.org)

statsmodels 是一个用于 Python 的库，正如你可能从名称中猜出的那样，其让用户能够通过使用各种统计模型估计方法以及执行统计断言和分析来进行数据探索。

许多有用的特征是描述性的，并可通过使用线性回归模型、广义线性模型、离散选择模型、稳健的线性模型、时序分析模型、各种估计器进行统计。

该库还提供了广泛的绘图函数，专门用于统计分析和调整使用大数据统计数据的良好性能。

结论

这个列表中的库被很多数据科学家和工程师认为是最顶级的，了解和熟悉它们是很有价值的。这里有这些库在 GitHub 上活动的详细统计：

![](http://t10.baidu.com/it/u=2974696854,1306903635&fm=170&s=6852C41B172D590B5A5534DB0300C0B1&w=640&h=338&img.JPEG)

当然，这并不是一份完全详尽的列表，还有其它很多值得关注的库、工具包和框架。比如说用于特定任务的 SciKit 包，其中包括用于图像的 SciKit-Image。如果你也有好想法，不妨与我们分享。

