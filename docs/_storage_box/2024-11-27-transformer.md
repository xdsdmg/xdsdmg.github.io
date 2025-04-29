---
layout: post
title: "Transformer"
date: 2024-11-27 20:13:09 +0800
categories: draft 
---

<div align="center">
<img src="/assets/imgs/transformer/arch.png" width="60%"/>
</div>
<div align="center">
<span style="font-size: 14px">图 1：Transformer 模型架构</span>
</div>

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
