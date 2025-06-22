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

Transformer 是一种主要应用于自然语言处理（NLP）领域的 seq2seq 模型。在深入理解 Transformer 之前，首先需要了解 NLP 任务中如何将文本转换为可计算的数学向量，这一步骤是所有 NLP 模型的基础和关键。

> 注：<br> **seq2seq 模型**：在 NLP 中，seq2seq（Sequence-to-Sequence）泛指一类模型，这类模型的核心思想是将输入序列映射为输出序列，两者长度可以不同。

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

### **2. Transformer 诞生的历史背景**

> [《Attention Is All You Need》](https://arxiv.org/pdf/1706.03762) 摘要：<br> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an **attention mechanism**. <br>  <br> 翻译： <br> 主流的序列转换模型基于复杂的递归或卷积神经网络，其中包括一个编码器和一个解码器。性能最好的模型还通过注意力机制将编码器和解码器连接起来。

#### **注意力机制**

注意力机制并不是在深度学习领域诞生的，甚至最早可以追溯至上世纪 60 年代的统计学算法 [Nadaraya-Watson 核回归（1964）](https://en.wikipedia.org/wiki/Kernel_regression)。

- 不随意线索 CNN
- 随意线索 Attention

query 不随意线索，key 是随意线索

60 年代 Nadaraya-Watson 核回归

$$
f(x) = \sum^n_{i = 1} \frac{K(x - x_i)}{\sum^n_{j = 1} K(x - x_j)} y_i
$$

$K(x, x_i)$ 也称为注意力分数

$$
f(x) = \sum^n_{i = 1} {\rm {softmax}}(K(x, x_i)) y_i
$$

#### **Additive Attention**

$$
K({\bf k}, {\bf q}) = {\bf v}^T \tanh(W_k{\bf k} + W_q{\bf q})
$$

#### **Scaled Dot-Product Attention**

$$
K({\bf k}, {\bf q}) = \frac{1}{\sqrt{d}} {\bf k} \cdot {\bf q}
$$

#### **Encoder-Decoder 架构**

> [视频-61 编码器-解码器架构【动手学深度学习v2】](https://www.bilibili.com/video/BV1c54y1E7YP?spm_id_from=333.788.videopod.episodes&vd_source=30199bd82fc917072f79b98bb0ab9c36)


在 Transformer 诞生的那个年代（2017 年），NLP 领域中主流的 seq2seq 模型都采用 Encoder-Decoder 架构。这种架构的设计理念并不难理解，可以表示为如下形式：

$$
\begin{equation}
\begin{aligned}
H & = f_{\rm encoder}(X) \\
\mathbf{y}_t & = f_{\rm decoder}(H, [\mathbf{y}_1; \mathbf{y}_2; \dots ; \mathbf{y}_{t-1}])
\end{aligned}
\end{equation}
$$

其中，$f_{\rm encoder}$ 表示模型的 encoder 模块，$X = [\mathbf{x}\_1; \mathbf{x}\_2; \dots ; \mathbf{x}\_n]$ 表示模型的输入，$\mathbf{x}\_i$ 为第 $i$ 个 token 的向量表示，$f_{\rm encoder}$ 将输入 $X$ 转化为一种可被机器理解的隐藏（或中间）状态 $H = [\mathbf{h}\_1; \mathbf{h}\_2; \dots ; \mathbf{h}\_n]$，通常 $\mathbf{h}\_i$ 对应 $\mathbf{x}\_i$；$f_{\rm decoder}$ 表示模型的 decoder 模块，$f_{\rm decoder}$ 基于 $H$ 自回归（Auto-Regressive）生成结果，$\mathbf{y}\_i$ 表示生成的第 $i$ 个 token。

> 注：<br> **自回归（Auto-Regressive，AR）**：是一种序列生成方法，其核心特征是：当前时刻的输出仅依赖于过去时刻的生成结果，并通过逐步迭代的方式构造完整序列。从概率建模的角度，这一过程可形式化表示为联合概率的链式分解 $$P(\mathbf{y}_{1:n}) = P(\mathbf{y}_1) \prod^n_{t = 2} P(\mathbf{y}_{t} \mid \mathbf{y}_{1:t-1})$$。另外，自回归过程也可以看作马尔科夫链的广义形式。

举个例子，当我们进行中英翻译时，通常会先通读整个中文句子，形成整体理解（即建立"语义表征"），然后基于这个理解逐步生成英文表达。这个认知过程恰好对应了 encoder-decoder 架构的工作机制：  
- 编码阶段（$f_{\rm encoder}$）：通过阅读理解源语句，将其抽象为包含关键语义信息的中间表征（即您所说的"印象"）；
- 解码阶段（$f_{\rm decoder}$）：根据该语义表征，按目标语言规则逐步生成译文。

这种"先理解，再表达"的两阶段处理方式，正是现代机器翻译系统模仿人类翻译思维的核心设计。

对于 encoder 与 decoder，其实是按照功能对深度神经网络中的模块（或层）进行划分，通俗来讲，encoder 负责将输入转化为机器可理解的某种表示（中间状态），decoder 负责将这种表示转化为人类可理解的结果。

这里有个疑问，为什么有些模型无 encoder 与 decoder 一说？比如图像分类模型，以及什么样的模型适合使用 encoder 与 decoder？

深度神经网络可以视为一个非常复杂的函数，这个函数负责实现输入空间与输出空间之间的映射，当输出空间的结构较为简单或固定时，是无需特别在意 encoder 与 decoder 这些概念的，比如图像分类模型的输出仅仅是一个类型标签。但对于序列转换模型，它的输出长度是不确定的，输出空间较为复杂，实现思路通常是先将输入序列转化为内部表示，在基于内部表示自回归生成结果，且在生成过程中需考虑输入序列与输出序列的不同位置间的依赖（或关联）程度。

#### **Transformer 想解决什么问题**

当时主流的序列转换模型是基于 encoder-decoder 架构的 RNN 或 CNN。RNN 无法较好地实现并行化；CNN 无法较好地计算输入序列与输出序列的不同位置间的依赖程度，距离越远所需的计算量越大。

Transformer 主要想达到这两个目标：
1. 能够较好地支持并行化；
2. 能够很方便地计算输入序列与输出序列的不同位置间的依赖程度的。

### **注意力机制**



#### **Self-Attention**

![a](https://zh-v2.d2l.ai/_images/cnn-rnn-self-attention.svg)



### **Transformer 工作原理**

Transformer 模型

<div align="center">
<img src="/assets/imgs/transformer/arch.png" width="60%"/>
</div>
<div align="center">
<span style="font-size: 14px">图 1：Transformer 模型架构</span>
</div>

$W_Q \in \mathbb{R}^{d_{\rm{model}} \times d_k}$、$W_K \in \mathbb{R}^{d_{\rm{model}} \times d_k}$、$W_V \in \mathbb{R}^{d_{\rm{model}} \times d_v}$、$X \in \mathbb{R}^{d_l \times d_k}$、$X^T = [x_1; x_2; \dots ; x_{d_l} ]$

$$
Q = W_Q X^T = [q_1; q_2; \dots ; q_{d_l}] \in \mathbb{R}^{d_{\rm{model}} \times d_l}
$$

$$
K = W_K X^T \in \mathbb{R}^{d_{\rm{model}} \times d_l}
$$

$$
V = W_V X^T \in \mathbb{R}^{d_{\rm{model}} \times d_l}
$$

$$
{\rm Attention}(Q, K, V) = {\rm softmax}(\frac{Q^TK}{\sqrt{d_k}})V^T \in \mathbb{R}^{ d_l \times d_{\rm{model}}}
$$

$$
Z=f(X)
$$

$$
y_t = f(\{y_1，y_2,\cdots,y_{t-1}\},Z)
$$

#### **LayerNorm**

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