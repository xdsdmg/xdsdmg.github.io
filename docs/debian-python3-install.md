# Debian 通过编译源码安装 Python3

## 安装相关依赖

``` bash
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
       libbz2-dev libreadline-dev wget curl llvm \
       libncurses5-dev libncursesw5-dev xz-utils tk-dev gdebi-core \
       libffi-dev libgdbm-dev \
       sqlite3 libsqlite3-dev
```

## 配置 SSL

``` bash
wget https://www.openssl.org/source/openssl-1.1.1t.tar.gz

tar -zxvf openssl-1.1.1t.tar.gz -C /home/zhangchi/workarea/modules
cd /home/zhangchi/workarea/modules/openssl-1.1.1t

./config --prefix=/home/zhangchi/workarea/modules/ssl  

make -j32
make test
make install
```

## 安装 Python3

``` bash
./configure --prefix=/usr/local \
    --with-openssl=/home/zhangchi/workarea/modules/ssl \
    --with-openssl-rpath=/home/zhangchi/workarea/modules/ssl/lib \
    --enable-shared LDFLAGS="-Wl,-rpath /usr/local/lib" \
    --enable-optimizations

make -j32
make test
make install

alias py3="/usr/local/bin/python3"
alias pip="/usr/local/bin/pip3"
```
