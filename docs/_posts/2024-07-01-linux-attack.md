---
layout: post
title: "安全方面的杂谈"
date: 2024-07-01 20:13:09 +0800
categories: post
---

![cover](/assets/imgs/linux-attack.png)

### 引言

这两天在面试一个安全开发相关的岗位，为了准备面试，回顾了下自己与安全相关的工作经历，也重新学习了解了一些新东西，记录下自己比较感兴趣的一些点。

### 如何判断 Linux 系统是否被攻击？

虽然安全领域中有很多非常厉害的工具，但我想从实际问题出发，比如我有一个 Linux 服务器，我如何（初步）判断它是否被攻击呢？（注：我并不是专业的安全渗透工程师，下面是我个人粗浅的见解。)

我想到两个思路：

1. 查看服务器当前各项性能指标的数值与平常相比是否有异常；
2. 查看服务器的相关日志或配置文件是否有异常的记录。

对于第 1 点，要先了解或记录下服务器在正常情况下各项性能指标的数值作为基线数据，需要平时多留意观察，再用当前的数据与基线数据比较，看看是否有差别。前段时间轰动一时的 xz 漏洞就是因为 CPU 占用过高被注意到的<sup>[[1]](#ref-1)</sup>。

对于第 2 点，这个方向涉及的点会比较杂。

首先，可以查看下`/etc/passwd`或`/etc/shadow`文件以检查系统中是否有异常用户，使用`last`命令查看最近有哪些用户登录过服务器（注：`lastb`命令只展示登录失败的记录），以及检查系统中一些重要的服务配置文件是否有被篡改，比如 SSH。

可以检查下系统日志文件，看看其中是否有异常，具体可参考这篇文献<sup>[[2]](#ref-2)</sup>，下面是`/var/log/messages`的示例。

``` bash 
# 下面例子来自 ChatGPT，仅供参考

# 检查是否有未经授权的用户或进程试图修改系统文件或关键配置文件的权限
Jul  1 12:15:30 server sudo: unauthorized attempt to change file permissions on /etc/shadow by user attacker

# 观察是否有不寻常的网络连接或流量，特别是大量的数据传输或与未知 IP 地址的通信
Jul  1 14:20:10 server kernel: [UFW BLOCK] IN=eth0 OUT= MAC=... SRC=203.0.113.5 DST=192.168.1.10 LEN=40 TOS=0x00 PREC=0x00 TTL=54 ID=0 DF PROTO=TCP SPT=54321 DPT=22 WINDOW=0 RES=0x00 RST URGP=0

# 查看是否有未知的进程在系统中运行，或者是否有不寻常的文件操作（如删除、修改文件）
Jul  1 15:30:45 server kernel: [ 123.456789] audit: type=1400 audit(1234567890.123:456): apparmor="DENIED" operation="open" profile="/usr/sbin/sshd" name="/etc/shadow" pid=12345 comm="sshd" requested_mask="r" denied_mask="r" fsuid=0 ouid=0
```

也可以通过`netstat`命令查看当前系统中是否有异常网络连接，及`tcpdump`抓包检查是否有异常流量。

还可以看看`.bash_history`、`.zsh_history`等这些记录执行过的历史 shell 命令的文件中有没有奇怪的记录。

### 一些专有名词

安全领域有很多专有名词，记录下

- SIEM（Security Information and Event Management）
- SOAR（Security Orchestration, Automation, and Response）
- IDS（Intrusion Detection System）
- IPS（Intrusion Prevention System）
- DLP（Data Loss Prevention）
- UEBA（User and Entity Behavior Analytics）
- EPP（Endpoint Protection Platform）
- EDR（Endpoint Detection and Response）
- TI（Threat Intelligence）
- NTA（Network Traffic Analysis）
- XDR（Extended Detection and Response）
- RASP（Runtime Application Self-Protection）
- WAF（Web Application Firewall）
- IoC（Indicators of Compromise）
- HIDS（Host Intrusion Detection System）
- CWPP（Cloud Workload Protection Platform）<sup>[[3]](#ref-3)</sup>

这些技术似乎每一个都是非常复杂的领域，从不同的角度提高系统的安全水位。

有些技术侧重于完成下面这样一个链条（注：这个链条让我联想到攻击溯源技术<sup>[[4]](#ref-4)</sup>）：

1. 采集各种终端设备的日志记录；
2. 将这些数据标准化为一致的结构（比如：源 IP、目的 IP、事件类型、时间戳等），可能还会对数据进行富化，额外添加一些字段；
3. 确定各项指标的基线数据，作为分析时的参考；
4. 分析数据：
    - 将收集到的数据与基线数据比较，是否有明显差异；
    - 将收集到的数据与 IoC 对照，是否存在威胁迹象；
    - 系统中是否存在异常行为，比如存在大量的异常失败请求。
5. 建立自动化监控与报警。

### 参考文献 

1. <a id='ref-1' href='https://www.openwall.com/lists/oss-security/2024/03/29/4'>The email publishing the existence of the xz attack</a>
2. <a id='ref-2' href='https://linux.vbird.org/linux_basic/mandrake9/0570syslog.php'>认识与分析登录档</a>
3. <a id='ref-3' href='https://www.cloudflare.com/zh-cn/learning/cloud/what-is-cwpp/'>什么是云工作负载保护平台（CWPP）？</a>
4. <a id='ref-4' href='https://websec.readthedocs.io/zh/latest/defense/forensic.html#'>溯源分析</a>
