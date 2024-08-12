---
layout: post
title:  "Debian 通过编译源码安装 Python3"
date:   2024-02-01 12:05:09 +0800
categories: post
---

### 安装相关依赖

``` bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
       libbz2-dev libreadline-dev wget curl llvm \
       libncurses5-dev libncursesw5-dev xz-utils tk-dev gdebi-core \
       libffi-dev libgdbm-dev \
       sqlite3 libsqlite3-dev
```

### 配置 SSL

pip 会依赖 openssl，系统自带的 openssl 版本可能过低，导致安装好后无法使用 pip。

``` bash
wget https://www.openssl.org/source/openssl-1.1.1t.tar.gz

tar -zxvf openssl-1.1.1t.tar.gz -C $HOME/workarea/modules
cd $HOME/workarea/modules/openssl-1.1.1t

./config --prefix=$HOME/workarea/modules/ssl  

make -j32
make test
make install
```

### 安装 Python3

``` bash
# 选择合适的版本
wget https://www.python.org/ftp/python/3.12.5/Python-3.12.5.tgz

./configure --prefix=/usr/local \
    --with-openssl=$HOME/workarea/modules/ssl \
    --with-openssl-rpath=$HOME/workarea/modules/ssl/lib \
    --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib" \
    --enable-optimizations

make -j32
make test
make install

alias py3="/usr/local/bin/python3"
alias pip="/usr/local/bin/pip3"
```
