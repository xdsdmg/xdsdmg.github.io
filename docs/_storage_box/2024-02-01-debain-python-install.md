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

[Python 源码下载列表](https://www.python.org/downloads/source/)

``` bash
# 选择合适的版本
wget https://www.python.org/ftp/python/3.12.5/Python-3.12.5.tgz

tar -zxvf Python3-3.12.5.tgz -C $HOME/workarea/modules

cd $HOME/workarea/modules

./configure --prefix=/usr/local \
    --with-openssl=$HOME/workarea/modules/ssl \
    --with-openssl-rpath=$HOME/workarea/modules/ssl/lib \
    --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib" \
    --enable-optimizations

make -j32
make test
make install
```

可以再设置下 alias

``` bash
alias py3="/usr/local/bin/python3"
alias pip="/usr/local/bin/pip3"
```

### 卸载 Python3

``` bash
# 清理 /usr/local/bin 目录
sudo rm /usr/local/bin/2to3*
sudo rm /usr/local/bin/pip*
sudo rm /usr/local/bin/pydoc3*
sudo rm /usr/local/bin/python*
sudo rm /usr/local/bin/idle*

# 清理 /usr/local/include 目录
sudo rm -rf /usr/local/include/python3.12

# 清理 /usr/local/lib 目录
sudo rm /usr/local/lib/libpython3*
sudo rm /usr/local/lib/python3*
sudo rm -rf /usr/local/lib/python3*
sudo rm -rf /usr/local/lib/pkgconfig
```

