---
layout: post
title: "Transformer"
date: 2024-11-27 20:13:09 +0800
categories: draft 
---

### **1. 引言**

~~自己在去年 11 月份学习了 [Transformer 模型](https://arxiv.org/abs/1706.03762)，并根据一些不错的 blog 实现了下此模型，后来几个月慢慢更深入地学习了深度学习相关知识，了解了更多，也慢慢发现 Transformer 中有些细节自己并不明白，比如：为什么 Tranformer 模型要分为 encoder 和 decoder 两个模块？为什么如今几乎所有的 LLM 模型都使用 decoder-Only 架构？最近两天又重新研究了下 Transformer 模型，写这篇 blog 记录一下。~~

[Transformer](https://arxiv.org/abs/1706.03762) 对当下 AI 领域的影响极其深远。目前，几乎所有大型语言模型 (LLM) 的架构都是在 Transformer 基础之上演进发展而来。不仅如此，支撑 LLM 运行的推理引擎，乃至底层硬件设计，都会专门考虑如何高效实现 Transformer 架构。因此，想要深入理解当下的 AI 领域，Tranformer 是非常值得研究的。

本次分享将按以下脉络展开：
1. 前置基础：文本是如何转化为可计算的数学向量的？
2. Transformer 诞生的历史背景，它旨在解决哪些核心问题？
3. Transformer 技术原理的通俗解释。
4. Transformer 在后续 LLM 中的演进，架构发生了哪些关键变化？
5. 关于 Transformer 架构的延伸讨论。

### **2. 如何将文本转化为可计算的数学向量**

> 计算机是如何对文字进行计算的呢？文字是无法直接参与数学计算的，计算机究竟对文字进行了哪些处理？

Transformer 是一种主要应用于自然语言处理（NLP）领域的 Seq2Seq 模型。在深入理解 Transformer 之前，首先需要了解 NLP 任务中如何将文本转换为可计算的数学向量，这一步骤是所有 NLP 模型的基础和关键。

> 注：<br> **Seq2Seq 模型**：在 NLP 中，Seq2Seq（Sequence-to-Sequence）泛指一类模型，这类模型的核心思想是将输入序列映射为输出序列，两者长度可以不同。

计算机通过分词（Tokenization）与词嵌入（Embedding）两个步骤对文本进行预处理，将其转化为可计算的数学向量。

#### **2.1 Tokenization**

Tokenization 的核心任务是将连续的自然语言文本按照语义或语法规则切分成独立的词语单元（token）。例如，对于输入文本"今天天气怎么样？"，Tokenizer 可能将其分解为 token 序列 ["今天", "天气", "怎么样", "？"]。  

目前 LLM（如 DeepSeek V3）普遍采用 [Byte-level BPE（BBPE）分词算法](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1e9441bbad598e181896349757b82af42b6a6902)，该算法在字节级别构建词表，能更好地处理多语言和特殊字符。  

在实际应用中，文本归一化（Normalization）是分词前的关键预处理步骤，目的是将不同形式、书写习惯或字符表示的文本统一为一致的格式，以避免因表面形式的差异导致分词错误。  

推荐材料：
1. 想进一步了解各种 Tokenization 算法可阅读这篇[文章](https://zhuanlan.zhihu.com/p/652520262)；
2. 如果对于 Tokenizer 的实现感兴趣，可以看看 Hugging Face 提供的 Tokenizer 的 Rust 实现（[代码仓库](https://github.com/huggingface/tokenizers)）。

#### **2.2 Embedding**

Embedding 的核心任务是将每个 token 映射为一个稠密向量（Dense Vector）。例如，词语“今天”可能被转化为向量$[1.1, 1.2, 2.0]$。

为什么叫“词嵌入”（Embedding）而不是“向量化”？
这一术语的起源可以追溯到 2013 年 Google AI 提出的 Word2vec 模型。“词嵌入”不仅实现了简单的向量化，还蕴含了更深层的设计理念：
1. 语义编码：语义相近的词，其向量在空间中的距离较近；语义无关的词，向量距离较远。
2. 上下文适应性：同一个词在不同语境下可能对应不同的向量表示（如多义词），从而承载更丰富的语义信息。
3. 高维表征：为了捕捉复杂的语义关系，词向量通常是多维的（几十维到几百维），远高于简单的“向量化”所暗示的数学操作。

因此，“词嵌入”强调的是将离散的词语嵌入到连续的向量空间中，并保留其语义关联，而不仅仅是形式上的向量转换。

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
<img src="/assets/imgs/transformer/arch.png" width="60%"/>
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
3. **特定语义**，比如“这个**苹果**很好吃”与“新发布的**苹果**手机”，两者中的“苹果”语义是不同的。

这种架构设计使得模型能够从不同的角度协同学习多样化的特征表示，从而更全面地建模序列内部的复杂关系。

##### **3.1.2 计算方法**

假设 $\mathbf{t}_i$ 表示第 $i$ 个 token，$j$ 表示第 $j$ 个注意力头。

首先，对每个输入token进行多组线性投影：

$$
\mathbf{q}_{i, j} = W^Q_j \mathbf{t}_i
$$

$$
\mathbf{k}_{i, j} = W^K_j \mathbf{t}_i
$$

$$
\mathbf{v}_{i, j} = W^V_j \mathbf{t}_i
$$

然后，在每个头上计算缩放点积注意力：

$$
\mathbf{o}_{i, j} = \sum^n_{k=1} {\rm softmax}(\frac{\mathbf{q}_{i, j}^T \mathbf{k}_{k, j}}{\sqrt{d}})\mathbf{v}_{k, j}
$$

最后，将所有头的输出拼接并通过线性变换：

$$
\mathbf{u}_i = W^O[\mathbf{o}_{i, 1}; \mathbf{o}_{i, 2} ; \dots ; \mathbf{o}_{i, n_h}]
$$

这种设计使得模型能够：

1. 同时关注不同位置的表示子空间
2. 学习更丰富的上下文依赖关系
3. 提高模型的泛化能力

#### **3.2 位置编码**

由于自注意力机制本身是位置无关的（permutation invariant），为了保留序列的顺序信息，Transformer 引入了位置编码。具体实现是将位置信息通过正弦/余弦函数编码后直接加到输入嵌入中：

$$
t_i = t_i + {\rm PE}(i)
$$

其中位置编码函数的设计满足：
- 能唯一标识每个位置
- 能处理比训练时更长的序列
- 有稳定的梯度传播特性

#### **Add & Norm**

Transformer 在每个子层（自注意力层和前馈网络层）后都采用了：
1. 残差连接（Add）：将子层输入与输出相加，缓解深层网络梯度消失问题
2. 层归一化（Norm）：对特征维度进行归一化，稳定训练过程

这种设计使得深层Transformer模型能够稳定训练，是模型成功的关键因素之一。

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