---
layout: post
title: "关于并发（WIP）"
date: 2024-07-05 23:55:09 +0800
categories: post
---

### 前言

最近看完了 [OSTEP](https://pages.cs.wisc.edu/~remzi/OSTEP/#book-chapters) 的并发部分，写这篇文章作为回顾。

现代操作系统可能是最早当然也是最典型的并发程序，现代操作系统需要协调多个任务（程序）的并发执行。

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
