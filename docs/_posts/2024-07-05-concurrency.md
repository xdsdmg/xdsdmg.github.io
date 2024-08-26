---
layout: post
title: "关于并发（WIP）"
date: 2024-07-05 23:55:09 +0800
categories: post
---

### 前言

最近看完了 [OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/#book-chapters) 的并发部分，想写一篇博客作为回顾，并发对我而言是一个很大很深奥的话题，甚至不知道该从何写起，这篇博客的逻辑可能会比较乱。在计算机科学领域，我们通常认为并发，要么是多个任务[^task]被分配到了不同的 CPU 上，这些任务同时运行；要么是多个任务被同一个 CPU 交替执行，由于交替的速度很快，这些任务看起来似乎是同时运行的。

由于现代操作系统需要协调多个任务的并发执行，现代操作系统可能是最早当然也是最典型的并发程序。对于操作系统的理论研究者，并发绝对是一个绕不开的话题。

这篇博客主要从 Linux 和 C 语言的角度，阐释一些并发相关的知识。

[^task]: 文中的任务是一个统称，指具有独立执行点的单位，可以是进程、线程或者协程等。

### 困惑

工作之后，我也经常会编写一些并发程序，但一直对并发程序背后的原理有一些困惑，主要有以下几个方面：

1. 并发程序的内存结构是怎样的？比如线程的栈内存是如何分配的？
2. 这样的程序会被编译成怎样的汇编代码？
3. 操作系统和 CPU 是如何交互的，即操作系统是如何让多个任务被同一个 CPU 交替执行的？或者是如何将多个任务分配到不同的 CPU 的？

#### 线程的内存分布

在 stack overflow 看到一则关于线程内存分布的回答（[链接](https://stackoverflow.com/a/54047901/24743435)），感觉很棒。大概意思是讲，我们可以把进程视为一个持有操作系统资源（比如，套接字）且相互隔离的容器，线程能够在这个容器中运行，并且与这个容器内的其他线程共享这个容器中的资源，线程才是 CPU 执行的基本单元，对于早期 UNIX 系统，在一个进程中只会有一个线程（主线程，在 C 语言中，可以认为是 main 函数的执行体），可以称为 1:1 模型，对于现代 Linux 系统，在一个进程中可以有多个线程，可以称为 1:N 模型。这里体现了操作系统的一些理念，引入进程概念的重要原因之一是资源隔离，而线程是为了提高程序的运行效率[^other]。

[^other]: 想起之前看到过这样一句话，运维的本质在于资源的隔离，不在于资源的扩充。

线程的内存分布基本如下图所示[^thread-mem-layout]，其中粉色区域为每个线程的栈空间，黄色区域为所有线程共享。ASLR 会影响内存空间的分布。


[^thread-mem-layout]: 主要参考 [CS:APP](https://csapp.cs.cmu.edu/) 9.7.2 小节与 [OSTEP 26 节第 2 页](https://pages.cs.wisc.edu/~remzi/OSTEP/threads-intro.pdf)，对于图中细节可参阅这两部分。


<div align="center">
<img src="/assets/imgs/thread-mem-layout.png" width="80%"/>
</div>

<p></p>

在 Linux 内核代码中，task_struct 结构体表示任务（进程或线程），也就是我们常讲的 PCB 或 TCB，对于版本 4.19.307，这个结构体定义在 include/linux/sched.h 597 行，当我们使用调用系统 API 创建一个线程时，实际会创建一个 task_struct 结构体，并放入操作系统的任务队列中，由调度器分配 CPU 资源执行。

以 x86 架构为例，通过执行 int 指令（软件中断）或 syscall（快速系统调用）指令来触发这种模式切换。

1. 操作系统与硬件是如何交互的？比如 CPU、内存及磁盘等。
2. 多线程程序的原理是怎样的？比如说汇编代码是怎样的，线程是如何成为独立的执行点的？
3. 在计算机中，操作系统能够维护自己的主导地位是由于时钟中断的存在，每隔一段时间，会触发时钟中断，执行中断处理程序，操作系统会再次拿到控制权。

序言章节给出了一个很形象的并发例子，假想面前有一张桌子，这张桌子上摆着一些桃子，每个人可以拿一个，这时有两种拿桃子的方式：

1. 大家一起拿，先在心里想好要拿哪一个再去拿，有可能想要拿的桃子先被其他人拿走了；
2. 大家排成一对，按顺序拿桃子，这种方式不会出现争抢的问题，但比较慢。

``` c
/// How to Use Inline Assembly Language in C Code?
///
/// https://gcc.gnu.org/onlinedocs/gcc/extensions-to-the-c-language-family/how-to-use-inline-assembly-language-in-c-code.html
/// https://stackoverflow.com/questions/71625166/trying-to-implement-a-spin-lock-via-lock-xchg-assembly

#include "common_threads.h"
#include <bits/pthreadtypes.h>
#include <stdio.h>
#include <string.h>

#define LOCK 1
#define UNLOCK 0

typedef volatile int vint;

static vint counter = 0;
static vint lock_t = 0;

int cmpxchgl(int expected, vint *status, int new_value) {
  asm volatile("lock cmpxchgl %2, %1"
               : "+a"(expected)
               : "m"(*status), "r"(new_value)
               : "memory", "cc");

  return expected;
}

int xchg(vint *addr, int newval) {
  int result;

  asm volatile("lock xchg %1, %0"
               : "=r"(result), "=m"(*addr)
               : "0"(newval), "m"(*addr)
               : "memory");

  return result;
}

void lock_cmpxchgl(vint *lock) {
  while (cmpxchgl(UNLOCK, lock, LOCK) != UNLOCK)
    ;
}

void unlock_cmpxchgl(vint *lock) {
  asm volatile("movl %1, %0" : "=m"(*lock) : "r"(UNLOCK) : "memory");
}

void lock_xchg(vint *lock) {
  while (xchg(lock, LOCK))
    ;
}

void unlock_xchg(vint *lock) { xchg(lock, UNLOCK); }

void *mythread(void *arg) {
  printf("%s: begin\n", (char *)arg);

  for (int i = 0; i < 1e7; i++) {
    lock_cmpxchgl(&lock_t);
    counter++;
    unlock_cmpxchgl(&lock_t);
  }

  printf("%s: end\n", (char *)arg);

  return NULL;
}

int main(int argc, char *argv[]) {
  if (argc > 1 && strcmp(argv[1], "test_xchg") == 0) {
    lock_xchg(&lock_t);
    printf("lock: lock_t: %d\n", lock_t);

    unlock_xchg(&lock_t);
    printf("unlock: lock_t: %d\n", lock_t);

    return 0;
  }

  if (argc > 1 && strcmp(argv[1], "test_cmpxchgl") == 0) {
    lock_cmpxchgl(&lock_t);
    printf("lock: lock_t: %d\n", lock_t);

    unlock_cmpxchgl(&lock_t);
    printf("unlock: lock_t: %d\n", lock_t);

    return 0;
  }

  pthread_t p1, p2;

  printf("main: begin (counter = %d)\n", counter);

  Pthread_create(&p1, NULL, mythread, "A");
  Pthread_create(&p2, NULL, mythread, "B");

  Pthread_join(p1, NULL);
  Pthread_join(p2, NULL);

  printf("main: end (counter = %d)\n", counter);

  return 0;

}
```
