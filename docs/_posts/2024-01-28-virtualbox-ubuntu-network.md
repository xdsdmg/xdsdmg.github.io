---
layout: post
title:  "VirtualBox Ubuntu 22.04 配置网络"
date:   2024-01-28 15:05:09 +0800
categories: jekyll update
---

### Ubuntu 网络配置参考链接

- [How To Change Network Interface Priority in Ubuntu?](https://devicetests.com/change-network-interface-priority-ubuntu)
- [Linux 路由表](https://www.jianshu.com/p/8499b53eb0a5)
- [How can I configure default route metric with dhcp and netplan?](https://askubuntu.com/questions/1008571/how-can-i-configure-default-route-metric-with-dhcp-and-netplan)

### 配置步骤

创建 Host-Only 和 NAT 网络，其中 Host-Only 网卡用于宿主机连接虚拟机，NAT 网卡用于虚拟机连接外部网络。

![img](/assets/1704634136220.jpg)

进入虚拟机执行以下命令更改网络配置文件。

``` bash
vi /etc/netplan/00-installer-config.yaml
```

网络配置文件如下，其中 enp0s3 为 NAT 网卡，enp0s8 为 Host-Only 网卡，这里需要设置网卡的优先级，将 enp0s8 的 metric 设置为 110（metric 越大优先级越低，默认为 100），否则 enp0s8 会与 enp0s3 冲突，影响虚拟机访问外部网络。

``` yaml
# This is the network config written by 'subiquity'
network:
  ethernets:
    enp0s3:
      dhcp4: true
    enp0s8:
      dhcp4: false
      addresses:
        - 192.168.56.10/24
      routes:
        - to: 0.0.0.0/0
          via: 192.168.56.1
          metric: 110
      nameservers:
        addresses:
          - 8.8.8.8
          - 8.8.4.4
  version: 2
```
