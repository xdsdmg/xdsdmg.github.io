---
layout: post
title: "Transformer"
date: 2024-11-27 20:13:09 +0800
categories: draft 
---

### **引言**

自己在去年 11 月份学习了 [Transformer 模型](https://arxiv.org/abs/1706.03762)，并根据一些不错的 blog 实现了下此模型，后来几个月慢慢更深入地学习了深度学习相关知识，了解了更多，也慢慢发现 Transformer 中有些细节自己并不明白，比如：为什么 Tranformer 模型要分为 encoder 和 decoder 两个模块？为什么如今几乎所有的 LLM 模型都使用 decoder-Only 架构？最近两天又重新研究了下 Transformer 模型，写这篇 blog 记录一下。

### **Transformer 的诞生背景**

> [《Attention Is All You Need》](https://arxiv.org/pdf/1706.03762) 摘要：<br> The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an **attention mechanism**.

#### **Encoder 与 Decoder**

> [视频-61 编码器-解码器架构【动手学深度学习v2】](https://www.bilibili.com/video/BV1c54y1E7YP?spm_id_from=333.788.videopod.episodes&vd_source=30199bd82fc917072f79b98bb0ab9c36)

对于 encoder 与 decoder，其实是按照功能对深度神经网络中的模块（或层）进行划分，通俗来讲，encoder 负责将输入转化为机器可理解的某种表示（中间状态），decoder 负责将这种表示转化为人类可理解的结果。

这里有个疑问，为什么有些模型无 encoder 与 decoder 一说？比如图像分类模型，以及什么样的模型适合使用 encoder 与 decoder？

深度神经网络可以视为一个非常复杂的函数，这个函数负责实现输入空间与输出空间之间的映射，当输出空间的结构较为简单或固定时，是无需特别在意 encoder 与 decoder 这些概念的，比如图像分类模型的输出仅仅是一个类型标签。但对于序列转换模型，它的输出长度是不确定的，输出空间较为复杂，实现思路通常是先将输入序列转化为内部表示，在基于内部表示自回归生成结果，且在生成过程中需考虑输入序列与输出序列的不同位置间的依赖（或关联）程度。

#### **Transformer 想解决什么问题**

当时主流的序列转换模型是基于 encoder-decoder 架构的 RNN 或 CNN。RNN 无法较好地实现并行化；CNN 无法较好地计算输入序列与输出序列的不同位置间的依赖程度，距离越远所需的计算量越大。

Transformer 主要想达到这两个目标：
1. 能够较好地支持并行化；
2. 能够很方便地计算输入序列与输出序列的不同位置间的依赖程度的。

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
