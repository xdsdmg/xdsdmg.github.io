---
layout: post
title: "聊一聊 Transformer"
date: 2024-11-27 20:13:09 +0800
categories: draft 
---

### **1. 引言**

[Transformer](https://arxiv.org/abs/1706.03762) 在 AI 的发展中称得上是一种**划时代**的技术。目前，几乎所有大语言模型 (LLM) 的模型架构均是在 Transformer 架构的基础上演变而来。另外，现代 AI 推理引擎（软件层面）的核心优化目标，以及专用 AI 加速硬件（硬件层面）的设计理念，都会考量 Transformer 的结构特性。因此，想要深入理解当下的 AI 领域，Transformer 是非常值得研究的。

> 注：<br>算法与硬件相互考量的 co-design 设计可能是未来的趋势。<br>推荐阅读：[Insights into DeepSeek-V3: Scaling Challenges and Reflections on
Hardware for AI Architectures](https://arxiv.org/pdf/2505.09343)

这篇文章希望能为大家展示 Transformer 的前因与后果，不局限于 Transformer 的技术原理，而是展示这一段技术演变的历史过程，**最大的目标是能让大家感受到一些 AI 之美**。

本次分享将按以下脉络展开：
1. **前置基础**：文本是如何转化为可计算的数学向量的？
2. **历史背景**：Transformer 诞生的历史背景，它旨在解决哪些核心问题？
3. **技术原理**：Transformer 技术原理的通俗解释。
4. **后续演进**：Transformer 在后续 LLM 中的演进，架构发生了哪些关键变化？

### **2. 如何将文本转化为可计算的数学向量**

> **关键问题**：<br>LLM 本质上是基于概率的数学建模系统，但自然语言符号本身不具备可计算性，所以计算机是如何对文字进行计算的呢？计算机究竟对文字进行了哪些处理？

Transformer 是一种主要应用于自然语言处理（Natural Language Processing，NLP）领域的 Seq2Seq 模型。在深入理解 Transformer 之前，首先需要了解 NLP 任务中如何将文本转换为可计算的数学向量，这一步骤是所有 NLP 模型的基础和关键。

> 注：<br> **Seq2Seq 模型**：在 NLP 中，Seq2Seq（Sequence-to-Sequence）泛指一类模型，这类模型的核心思想是将输入序列映射为输出序列，两者长度可以不同。

计算机通过分词（Tokenization）与词嵌入（Word Embedding）两个步骤对文本进行预处理，将其转化为可计算的数学向量。

#### **2.1 分词（Tokenization）**

**Tokenization 的核心任务是将连续的自然语言文本按照语义或语法规则切分成独立的词语单元（token）。**执行分词这个动作的模块被称为**分词器（Tokenizer）**，例如，对于输入文本“今天天气怎么样？”，Tokenizer 将其分解为 token 序列 ["今天", "天气", "怎么样", "？"]。

那 tokenizer 是如何构建和工作的呢？
1. **构建（训练阶段）**：通过大量的训练文本构建词表；
2. **工作（推理阶段）**：tokenizer 基于词表对输入文本进行分词。

目前 LLM（如 DeepSeek V3）普遍采用 **Byte-level [Byte Pair Encoding](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1e9441bbad598e181896349757b82af42b6a6902)（BBPE）**分词算法，该算法在字节级别构建词表，能更好地处理多语言和特殊字符。 

> Byte Pair Encoding 算法的逻辑其实非常简单，大家如果感兴趣可以看[论文](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1e9441bbad598e181896349757b82af42b6a6902)的第 4 页。<br><br>**Find the most frequent pair of consecutive two character codes in the text, and then substitute an unused code for the occurrences of the pair.**

在实际应用中，文本归一化（Text Normalization）是构建词表时的关键预处理步骤，目的是将不同形式、书写习惯或字符表示的文本统一为一致的格式，以避免因表面形式的差异导致分词错误。  

1. **形式统一化**：消除书写变体（如全角/半角字符）、拼写差异（如美式/英式英语）和字符编码差异（如 Unicode 组合字符），比如“colour”（英式）与“color”（美式）、Unicode 编码的 NFKC 标准化处理；
2. **语义一致性**：确保相同语义的文本单元获得相同的向量表示，比如“café”与“cafe”、“I'm”与“I am”；
3. **词表效率**：通过减少表面形式的多样性，提升词表的空间利用率。

> 如果平常大家有使用 Hugging Face 下载 LLM 文件，会发现每个模型仓库里基本都会有一个`tokenizer.json`文件，这个文件存储了 tokenizer 完整的配置信息，当然也包括词表数据。 

>推荐材料：
1. 如果想进一步了解各种词表构建算法，可阅读这篇[文章](https://zhuanlan.zhihu.com/p/652520262)；
2. 如果对于 Tokenizer 的实现感兴趣，可以看看 Hugging Face 提供的 Tokenizer 的 Rust 实现（[代码仓库](https://github.com/huggingface/tokenizers)）。

#### **2.2 词嵌入（Word Embedding）**

**词嵌入（Word Embedding）的核心任务是将 tokenizer 输出的每个 token 映射为一个稠密向量（Dense Vector）。**例如，token“今天”可能被转化为向量 $[1.1, 1.2, 2.0]$。

为什么叫“词嵌入”（Embedding）而不是“向量化”？
这一术语的起源可以追溯到 2013 年 Google AI 提出的 [Word2vec](https://arxiv.org/abs/1301.3781) 模型。词嵌入不仅实现了简单的向量化，还蕴含了更深层的设计理念：
1. **语义编码**：语义相近的词，其向量在空间中的距离较近；语义无关的词，向量距离较远。
2. **上下文适应性**：同一个词在不同语境下可能对应不同的向量表示（如多义词），从而承载更丰富的语义信息。
3. **高维表征**：为了捕捉复杂的语义关系，词向量通常是多维的（几十维到几百维），远高于简单的“向量化”所暗示的数学操作。

因此，**词嵌入强调的是将离散的词语嵌入到连续的向量空间中，并保留其语义关联**，而不仅仅是形式上的向量转换。

推荐阅读：[没有思考过 Embedding，不足以谈 AI]( https://zhuanlan.zhihu.com/p/643560252)

### **3. Transformer 诞生的历史背景**

> [《Attention Is All You Need》](https://arxiv.org/pdf/1706.03762) 摘要：<br> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an **attention mechanism**. <br>  <br> 翻译： <br> 主流的序列转换模型基于复杂的递归或卷积神经网络，其中包括一个编码器和一个解码器。性能最好的模型还通过注意力机制将编码器和解码器连接起来。

#### **3.1 Encoder-Decoder 架构**

> 1. 什么是 Encoder-Decoder 架构？
> 2. 为什么主流的 Seq2Seq 模型都采用 Encoder-Decoder 架构？

在 Transformer 诞生的那个年代（2017 年），NLP 领域中主流的 Seq2Seq 模型都采用 Encoder-Decoder 架构。这种架构的设计理念并不难理解，可以表示为如下形式：

$$
\begin{equation}
\begin{aligned}
H & = f_{\rm encoder}(X) \\
\mathbf{y}_t & = f_{\rm decoder}(H, [\mathbf{y}_1; \mathbf{y}_2; \dots ; \mathbf{y}_{t-1}])
\end{aligned}
\end{equation}
$$

其中，$f_{\rm encoder}(\cdot)$ 表示模型的 encoder 模块，$X = [\mathbf{x}\_1; \mathbf{x}\_2; \dots ; \mathbf{x}\_n]$ 表示模型的输入，$\mathbf{x}\_i$ 为第 $i$ 个 token 的向量表示，$f_{\rm encoder}(\cdot)$ 将输入 $X$ 转化为一种可被机器理解的隐藏（或中间）状态 $H$；$f_{\rm decoder}(\cdot, \cdot)$ 表示模型的 decoder 模块，$f_{\rm decoder}(\cdot, \cdot)$ 基于 $H$ 自回归（Auto-Regressive）生成结果，$\mathbf{y}\_i$ 表示生成的第 $i$ 个 token。

> 注：<br> **自回归（Auto-Regressive，AR）**：是一种序列生成方法，其核心特征是：当前时刻的输出仅依赖于过去时刻的生成结果，并通过逐步迭代的方式构造完整序列。从概率建模的角度，这一过程可形式化表示为联合概率的链式分解 $$P(\mathbf{y}_{1:n}) = P(\mathbf{y}_1) \prod^n_{t = 2} P(\mathbf{y}_{t} \mid \mathbf{y}_{1:t-1})$$。另外，自回归过程也可以看作马尔科夫链的广义形式。

举个例子，当我们进行中英翻译时，通常会先通读整个中文句子，形成整体理解（即建立"语义表征"），然后基于这个理解逐步生成英文表达。这个认知过程恰好对应了 encoder-decoder 架构的工作机制：  
- 编码阶段（$f_{\rm encoder}(\cdot)$）：通过阅读理解源语句，将其抽象为包含关键语义信息的中间表征（即您所说的"印象"）；
- 解码阶段（$f_{\rm decoder}(\cdot, \cdot)$）：根据该语义表征，按目标语言规则逐步生成译文。

这种“先理解，再表达”的两阶段处理方式，正是现代机器翻译系统模仿人类翻译思维的核心设计。

对于 encoder 与 decoder，其实是按照功能对深度神经网络中的模块（或层）进行划分，通俗来讲，encoder 负责将输入转化为机器可理解的某种表示（中间状态），decoder 负责将这种表示转化为人类可理解的结果。

这里有个疑问，为什么有些模型无 encoder 与 decoder 一说？比如图像分类模型，以及什么样的模型适合使用 encoder 与 decoder？

深度神经网络可以视为一个非常复杂的函数，这个函数负责实现输入空间与输出空间之间的映射，当输出空间的结构较为简单或固定时，是无需特别在意 encoder 与 decoder 这些概念的，比如图像分类模型的输出仅仅是一个类型标签。但对于序列转换模型，它的输出长度是不确定的，输出空间较为复杂，实现思路通常是先将输入序列转化为内部表示，在基于内部表示自回归生成结果，且在生成过程中需考虑输入序列与输出序列的不同位置间的依赖（或关联）程度。

> 推荐资料：[视频-61 编码器-解码器架构【动手学深度学习v2】](https://www.bilibili.com/video/BV1c54y1E7YP?spm_id_from=333.788.videopod.episodes&vd_source=30199bd82fc917072f79b98bb0ab9c36)

#### **3.2 注意力机制**

##### **3.2.1 注意力机制能够为 Encoder-Decoder 架构带来哪些提升？**

在早期使用 RNN 或 CNN 构建的基于 Encoder-Decoder 架构的 Seq2Seq 模型中，自回归生成过程存在显著的上下文对齐局限性。具体表现为：解码器在生成目标序列的每一个 token 时，仅能隐式地依赖编码器输出的固定长度上下文表示（如 RNN 的最终隐藏状态或 CNN 的顶层特征），而无法显式建模当前生成位置与输入序列关键片段之间的动态关联关系。这种机制缺陷在 NLP 任务中会导致两类典型问题：
1. 长距离依赖丢失，尤其是当输入序列较长时，编码器的信息压缩瓶颈会削弱关键输入的保留；
2. 局部强相关忽略，例如在机器翻译任务中，生成目标语言动词时可能无法精准关联源语言中对应的谓语成分。

这一局限性的本质原因在于传统 Encoder-Decoder 架构的静态编码特性：编码阶段将变长输入序列压缩为固定维度的向量（Context Vector），而解码阶段缺乏对该向量的细粒度访问机制。直到注意力机制的引入，才通过计算解码器当前状态与编码器所有状态的动态权重（Alignment Scores）解决了这一问题。

##### **3.2.2 注意力机制的起源**

注意力机制的雏形并非源自深度学习领域，其理论根源可追溯至 20 世纪 60 年代的统计学习框架——[Nadaraya-Watson 核回归（1964）](https://en.wikipedia.org/wiki/Kernel_regression)。该模型的数学表述为：

$$
f(x) = \sum^n_{i=1} \alpha(x,x_i) y_i,\quad \alpha(x,x_i) = \frac{K(x - x_i)}{\sum^n_{j=1} K(x - x_j)}
$$

其中 $\alpha(x,x_i)$ 可视为一种原始形式的注意力权重。该模型通过核函数 $K(\cdot)$ 构建输入空间上的概率密度估计，给定查询值 $x$ 时，基于训练集 $\{(x_i,y_i)\}_{i=1}^n$ 计算条件期望 $E[y \| x]$。具体而言：

1. **相似度度量**：核函数 $K(x-x_i)$ 本质是衡量查询 $x$ 与键 $x_i$ 的相似性度量，符合注意力机制中 query-key 匹配的核心思想
2. **权重归一化**：通过 softmax-like 的分母项实现注意力权重的归一化（$\sum_i\alpha(x,x_i)=1$）
3. **上下文感知**：输出是值 $y_i$ 的加权平均，权重随查询动态变化，这与现代注意力机制完全一致

> 其实 Nadaraya-Watson 核回归和 KNN 算法也有些相似，可以看作是 KNN 算法的概率化表达。

##### **3.2.3 什么是注意力机制**

~~- 不随意线索 CNN~~
~~- 随意线索 Attention~~

~~query 不随意线索，key 是随意线索~~

注意力机制可视为 Nadaraya-Watson 核回归在神经网络中的推广形式，其核心创新在于：
1. 将原始核权重 $\alpha$ 重构为 softmax 归一化形式
2. 输入输出空间从标量推广到高维张量
3. 引入可学习的特征变换

其通用数学表述为：

$$
f(\mathbf{q}) = \sum^n_{i=1} \alpha(\mathbf{q}, \mathbf{k}_i)\mathbf{v}_i,\quad 
\alpha(\mathbf{q}, \mathbf{k}_i) = \mathrm{softmax}(K(\mathbf{q}, \mathbf{k}_i))
$$

其中，$\mathbf{q} \in \mathbb{R}^d$ 为查询向量（Query），$\mathbf{k}_i \in \mathbb{R}^d$ 为键向量（Key），$\mathbf{v}_i \in \mathbb{R}^m$ 为值向量（Value），$K(\cdot,\cdot)$ 为注意力评分函数。

现代注意力机制主要采用两种评分函数：

1. **加性注意力（Additive Attention）**：$K(\mathbf{q}, \mathbf{k}) = \mathbf{w}^T \sigma(\mathbf{W}_q\mathbf{q} + \mathbf{W}_k\mathbf{k})$
   - 优点：通过全连接层实现跨维度交互，激活函数 $\sigma$（通常为 $\tanh$）提供非线性表达能力；
   - 缺点：参数量大，计算复杂度高。

2. **缩放点积注意力（Scaled Dot-Product Attention）**：$K(\mathbf{q}, \mathbf{k}) = \mathbf{q}^T\mathbf{k} / \sqrt{d}$
   - 优点：计算高效，无需额外参数；
   - 关键改进：缩放因子 $\sqrt{d}$ 防止高维点积值过大导致梯度消失。

**发展趋势**：
- 加性注意力在早期 Seq2Seq 模型中表现优异；
- 缩放点积注意力因计算效率成为 Transformer 架构的标准配置；
- 大语言模型（LLMs）普遍采用多头缩放点积注意力，通过并行计算实现高效处理长序列。


#### **3.3 Transformer 想解决什么问题**

当时主流的序列转换模型是基于 encoder-decoder 架构的 RNN 或 CNN。RNN 无法较好地实现并行化；CNN 无法较好地计算输入序列与输出序列的不同位置间的依赖程度，距离越远所需的计算量越大。

Transformer 主要想达到这两个目标：
1. 能够较好地支持并行化；
2. 能够很方便地计算输入序列与输出序列的不同位置间的依赖程度的。

### **注意力机制**


#### **Self-Attention**

![a](https://zh-v2.d2l.ai/_images/cnn-rnn-self-attention.svg)



### **3. Transformer 工作原理**

Transformer 是一个完全基于注意力机制的 Seq2Seq 模型，其核心创新在于完全摒弃了 RNN 与 CNN，仅依靠自注意力机制和前馈神经网络（Feed Forward Network）构建编码器-解码器结构。这一突破性设计使得 Transformer 能够高效地建模长距离依赖关系，并实现并行化计算。正因如此，其论文的标题才命名为"Attention Is All You Need"。

图 1 展示了 Transformer 的整体架构，通过前文的铺垫，大家或许已对 Transformer 的架构设计有了一定感触。

<div align="center">
<img src="/assets/imgs/transformer/arch.png" width="70%"/>
</div>
<div align="center">
<span style="font-size: 14px">图 1：Transformer 模型架构</span>
</div>

接下来我们按 Transformer 模型的主要模块来介绍，主要有多头注意力机制、位置编码、前馈神经网络与 Add & Norm 这几个核心模块。

#### **3.1 多头注意力机制**

> 在深度学习中，“头”可以理解为独立、并行的计算单元。

##### **3.1.1 介绍**

Transformer 采用了多头注意力机制（Multi-Head Attention，MHA）而不是传统的注意力机制以增强模型的表达能力这种，机制可以理解为：

1. **并行计算**：将输入序列通过多个独立、并行的注意力机制算子（或注意力头）同时进行计算处理；
2. **模式学习**：每个注意力头学习不同的注意力模式，例如，有的注意力头擅长捕捉长距离依赖关系，有的关注局部语法特征，有的提取特定语义信息等；
3. **特征融合**：最后将所有注意力头的输出拼接，并通过线性变换得到最终结果。

下面举几个例子来做一些辅助说明：

1. **长距离依赖关系**，比如“The **cat** that wandered into our garden last winter **was** starving, but now is happily napping in the sun.”这段英文叙述，第 10 个词为“was”而不是“were”取决于第 2 个词“cat”。
2. **局部语法特征**，比如“**a** book”与“**an** apple”，根据名词的发音特征，前者使用“a”，后者使用“an”。
3. **特定语义**，比如“我的车旁边停了一辆剁椒鱼头”与“这家店的剁椒鱼头还不错”，两者中的“剁椒鱼头”语义是不同的。

这种架构设计使得模型能够从不同的角度协同学习多样化的特征表示，从而更全面地建模序列内部的复杂关系。

##### **3.1.2 计算方法**

前文有提及，对于将输入序列构建为内在表示（理解）与基于内在表示自回归生成结果（表达）两个过程，Transformer 皆使用注意力机制完成，而非传统的 RNN 与 CNN。
因为用于注意力机制（详见 3.2 小节）计算的 query、key、value 完全由输入 token 序列通过线性变换（乘以一个矩阵，这个矩阵是需要学习的参数）而来，所以这种设计模式被称为**自**注意力机制。

下面简单介绍一下计算过程，假设 $\mathbf{t}_i$ 表示第 $i$ 个 token（Embedding 表示），$j$ 表示第 $j$ 个注意力头，$\mathbf{t}_i$ 在第 $j$ 个注意力头中的 query、key 与 value 分别表示为 $\mathbf{q}\_{i, j}$、$\mathbf{k}\_{i, j}$ 与 $\mathbf{v}\_{i, j}$。

首先，按头对输入的每个 token $\mathbf{t}_i$ 进行线性变换，得到对应的 query、key 与 value：

$$
\mathbf{q}_{i, j} = W^Q_j \mathbf{t}_i
$$

$$
\mathbf{k}_{i, j} = W^K_j \mathbf{t}_i
$$

$$
\mathbf{v}_{i, j} = W^V_j \mathbf{t}_i
$$

其中，$t_i \in \mathbb{R}^{d_t}$，$W^Q_j \in \mathbb{R}^{d \times d_t}$，$W^K_j \in \mathbb{R}^{d \times d_t}$，$W^V_j \in \mathbb{R}^{d \times d_t}$。

然后，在每个头上计算缩放点积注意力：

$$
\mathbf{o}_{i, j} = \sum^n_{k=1} {\rm softmax}_k(\frac{\mathbf{q}_{i, j}^T \mathbf{k}_{k, j}}{\sqrt{d}}) \cdot \mathbf{v}_{k, j} = \sum^n_{k=1} \frac{\frac{e^{\mathbf{q}_{i, j}^T \mathbf{k}_{k, j}}}{\sqrt{d}}}{\sum^n_{m=1}\frac{e^{\mathbf{q}_{i, j}^T \mathbf{k}_{m, j}}}{\sqrt{d}}} \cdot \mathbf{v}_{k, j}
$$

> 注：${\rm softmax}$ 是一种在机器学习中广泛使用的算子，对于向量 $\mathbf{x} = [x_1, x_2, \dots ,x_n]$，${\rm softmax}\_i(\mathbf{x}) = \frac{e^{x_i}}{\sum^n_{k=1}e^{x_k}}$，它可以将 $\mathbf{x}$ 中的每个元素转化为一个 $0$ 至 $1$ 间的值，且所有元素转化后的值和为 $1$。

可以看出，对于第 $j$ 个头，其输出 $\mathbf{o}_{i, j}$ 为 $[\mathbf{v}\_{1,j}, \mathbf{v}\_{2,j},\dots,\mathbf{v}\_{n,j}]$ 的线性组合，可以简写为 $\mathbf{o}\_{i, j} = \sum^n\_{k=1} \alpha\_k \mathbf{v}\_{k,j}$，在机器学习中通常用向量的内积来衡量两个向量的相似度，$\mathbf{q}\_{i, j}$ 与 $\mathbf{k}\_{k, j}$ 的内积越大，或 $\mathbf{q}\_{i, j}$ 与 $\mathbf{k}\_{k, j}$ 越相似，$\mathbf{v}\_{k,j}$ 的权重 $\alpha_k$ 就会越大，$\mathbf{v}\_{k,j}$ 所占的比重就会越大。**简单来讲就是，与当前的词越相近的词，对结果的影响越大。**

最后，将所有头的输出拼接并通过线性变换：

$$
\mathbf{u}_i = W^O[\mathbf{o}_{i, 1}; \mathbf{o}_{i, 2} ; \dots ; \mathbf{o}_{i, n_h}]
$$

从图 1 可以看出，在 Transformer 中共有 3 个地方用到了多头注意力机制，第 1 个地方在 Encoder 模块，第 2、3 个地方在 Decoder 模块。
Encoder 中的多头注意力机制为普通的多头注意力机制（也称为双向注意力机制），
Decoder 底部的多头注意力机制（即 Masked Multi-Head Attention）为单向注意力机制（也称为因果注意力机制），Decoder 底部的多头注意力机制为交叉注意力机制。

这三种多头注意力机制的计算框架和上述是类似的，但有一些细微的差别。

###### **3.1.2.1 双向注意力机制**

Encoder-Decoder 架构的核心思想是“先理解后生成”。其中 Encoder 的核心任务是实现对输入的深度理解，这一过程通过全序列扫描（full-sequence reading）实现。具体表现为：当处理长度为 $l$ 的输入序列 $X = [\mathbf{t}_1, \mathbf{t}_2, \dots, \mathbf{t}_l]$ 时，每个位置的注意力计算会同时考虑序列中所有 token 的 key-value 对。因此公式(7)中的求和上限应修正为序列长度 $l$：

$$
\mathbf{o}_{i, j} = \sum^{\color{red} l}_{k=1} {\rm softmax}_k(\frac{\mathbf{q}_{i, j}^T \mathbf{k}_{k, j}}{\sqrt{d}}) \cdot \mathbf{v}_{k, j}
$$

这种全局可见的注意力模式使 Encoder 能够建立完整的上下文表征。

###### **3.1.2.2 单向注意力机制**

Decoder 负责基于 Encoder 生成的中间表示自回归生成结果，对于自回归过程，当前的输出只取决于之前的结果，与未来的结果无关。

Decoder 采用自回归方式生成输出，其核心约束是：当前时刻的预测只能依赖于已生成的输出（即历史信息）。这种时序依赖性通过注意力掩码（attention mask）实现，具体表现为：

1. **训练阶段**：  
   - 采用**因果掩码（causal mask）**，确保第 $i$ 个位置的 query 只能访问前 $i$ 个位置的 key-value 对。  
   - 这种掩码通常通过一个下三角矩阵（元素为 $-\infty$ 或 $0$）实现，使得 softmax 计算时未来位置的概率接近 $0$。  
   - 公式中的求和上限为当前位置 $i$，但实际实现时通常仍计算所有位置的注意力分数，再通过掩码屏蔽未来信息：  

   $$
   \mathbf{o}_{i, j} = \sum^{l}_{k=1} \text{mask}(k \leq i) \cdot {\rm softmax}_k\left(\frac{\mathbf{q}_{i, j}^T \mathbf{k}_{k, j}}{\sqrt{d}}\right) \cdot \mathbf{v}_{k, j}
   $$

2. **推理阶段**：  
   - 由于解码是逐步进行的（每次生成一个 token），模型只需计算当前 query 与历史 key-value 对的注意力，无需显式掩码。  
   - 为提升效率，通常会缓存（cache）历史 key-value 对，避免重复计算。 

$$
\mathbf{o}_{i, j} = \sum^{\color{red} i}_{k=1} {\rm softmax}_k(\frac{\mathbf{q}_{i, j}^T \mathbf{k}_{k, j}}{\sqrt{d}}) \cdot \mathbf{v}_{k, j}
$$

###### **3.1.2.3 交叉注意力机制**

在解码阶段，Decoder 需要将 Encoder 输出的上下文表征（通常记为 $\mathbf{U} = [\mathbf{u}_1, \mathbf{u}_2, \dots, \mathbf{u}_l]$）融入生成过程。交叉注意力机制的查询（query）来自 Decoder 的当前状态，而 key-value 对则来自 Encoder 的最终输出：

$$
\mathbf{o}_{i, j} = \sum^{\color{red} l}_{k=1} {\rm softmax}_k(\frac{\mathbf{q}_{i, j}^T {\mathbf{\color{red} u}_{\color{red} k}}}{\sqrt{d}}) \cdot \mathbf{\color{red} u}_{\color{red} k}
$$

该机制实现了 Encoder-Decoder 之间的信息桥接，是序列到序列建模的关键组件。

这种设计使得模型能够：

1. 同时关注不同位置的表示子空间
2. 学习更丰富的上下文依赖关系
3. 提高模型的泛化能力

#### **3.2 位置编码**

自注意力机制本身具有置换不变性（permutation invariant），这意味着其对输入序列的顺序不敏感。为了保留关键的序列位置信息，Transformer 引入了位置编码机制。该机制通过将位置信息与词嵌入向量相加来实现：

$$
\mathbf{t}_i = \mathbf{t}_i + \mathrm{PE}(i)
$$

其中位置编码函数 $\mathrm{PE}(i)$ 的设计遵循以下原则：
1. **唯一性**：每个位置具有独一无二的编码表示
2. **可扩展性**：能够处理超出训练时所见长度的序列
3. **稳定性**：具有良好的梯度传播特性，便于模型优化

位置编码采用正弦和余弦函数的组合形式，这种设计具有以下优势：
- 可以表示绝对位置信息
- 允许模型学习相对位置关系
- 能够自然地扩展到更长的序列长度

值得注意的是，由于自注意力机制的本质特性，相同的 Key-Value 对在不同位置与 Query 的计算结果理论上应该相同。然而，通过引入位置编码，Query 和 Key 的点积运算中自然地包含了位置角度信息，这使得模型能够区分不同位置的相同内容。

#### **Add & Norm**

Transformer 架构在每个子层（包括自注意力层和前馈网络层）后都采用了以下关键设计：

1. **残差连接（Residual Connection）**
   - 数学形式：$\mathbf{x} + \mathrm{Sublayer}(\mathbf{x})$
   - 功能作用：
     * 建立直接的梯度传播路径
     * 有效缓解深层网络的梯度消失问题
     * 使模型能够学习残差映射而非完整变换

2. **层归一化（Layer Normalization）**
   - 操作方式：沿特征维度进行归一化
   - 核心优势：
     * 稳定各层的输入分布
     * 加速模型收敛
     * 减少对初始化的敏感性

**领域差异说明**：
- 计算机视觉（CV）中常采用批归一化（BatchNorm）：沿批次维度归一化
- 自然语言处理（NLP）中偏好层归一化（LayerNorm）：沿特征维度归一化
（这种差异主要源于文本数据的变长特性与批处理挑战）

这种 Add & Norm 的组合设计：
- 使深层Transformer（如12层以上的模型）能够稳定训练
- 成为Transformer架构成功的关键因素之一
- 后续被多种神经网络架构广泛借鉴

对于$x\in\mathbb{R}^d$

$$
\mu = \frac{1}{d} \sum^d_{i=1} x_i
$$

$$
\sigma^2 = \frac{1}{d} \sum^d_{i=1} (x_i - \mu)^2
$$

$$
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
$$

{% 
assign eq =
'$$
\begin{bmatrix}
    Q_{1,1} & Q_{1,2} & \cdots & Q_{1,{\rm d\_model}} \\
    Q_{2,1} & Q_{2,2} & \cdots & Q_{2,{\rm d\_model}} \\
    \vdots  & \vdots  & \ddots & \vdots               \\
    Q_{{\rm len\_q},1} & Q_{1,2} & \cdots & Q_{1,{\rm d\_model}}
\end{bmatrix}
$$'
%}

{{eq}}

Collectively, these developments have ushered in an era where scaling model size and computational power is seen as the key to unlocking higher levels of intelligence.

只加大模型的尺寸没有被证明可以提升模型的推理能力（思维链）。

如何激发模型的能量？Rason Path

涌现能力，一种能力在较小的模型中不存在，但在较大的模型中存在。

大模型的发展历史



$$
L = - \mathbb{E}_{x \sim p(x)} \mathbb{E}_{y \sim \pi^* (\cdot|x)} [\log \pi (y|x)]
$$

#### **交叉熵**

$$
H(P, Q) = - \mathbb{E}_{y \sim P} \left[ \log Q(y) \right]
$$

#### **KL 散度**

$$
\begin{equation}
\begin{aligned}
D_{KL}(P \parallel Q)   &= \mathbb{E}_{y \sim P} \left[ \log \frac{P(y)}{Q(y)} \right]    \\ 
                        &= \mathbb{E}_{y \sim P} \left[\log P(y) - \log Q(y)\right]       \\
                        &= \mathbb{E}_{y \sim P} \left[\log P(y)\right] - \mathbb{E}_{y \sim P} \left[\log Q(y)\right] \\
                        &= - H(P) + H(P, Q) \\
                        &= H(P, Q) - H(P)
\end{aligned}
\end{equation}
$$