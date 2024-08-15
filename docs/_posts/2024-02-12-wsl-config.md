---
layout: post
title:  "Windows 与 WSL 配置"
date:   2024-02-12 14:05:09 +0800
categories: post
---

---

2024-08-11 买了台新电脑（铭凡 UM790PRO），又重新配置了下，更新部分内容。

---

### 安装 Debian

WSL 我一直使用 Debian，可以在 Micscoft Store 中直接安装，但第一次启动可能会报错，我当时重启下电脑就好了。

```
wslregisterdistribution failed with error: 0x8004032d
```

如果启动时有以下错误提示：

```
wsl: 检测到 localhost 代理配置，但未镜像到 WSL。NAT 模式下的 WSL 不支持 localhost 代理。
```

可以在 Windows 中`%USERPROFILE%\.wslconfig`文件添加下面内容：

```
[wsl2]
networkingMode=mirrored
dnsTunneling=true
firewall=true
autoProxy=true

[experimental]
# requires dnsTunneling but are also OPTIONAL
bestEffortDnsParsing=true
useWindowsDnsCache=true
```

配置下国内的软件源，提高 apt 或 apt-get 的下载速度。

[清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/help/debian/)

### 字体

编辑器、IDE 及终端等的字体我一直用 [Hack](https://github.com/source-foundry/Hack)（[下载链接](https://github.com/ryanoasis/nerd-fonts/releases/download/v3.2.1/Hack.zip)）这款字体。

### 编辑器配置

我平时使用 [Windows Terminal](https://github.com/microsoft/terminal) 作为终端开启 Debian，使用 Vim/[Neovim](https://neovim.io/) 与 [tmux](https://github.com/tmux/tmux/wiki) 的组合作为编辑器多一些。

#### Vim

Vim 的配置文件是`$HOME/.vimrc`，这是我最早的 Vim 配置，现在我的 Vim 只进行基础配置，不再安装额外的插件了，把这个脚本放在这里当作纪念。

``` vim
call plug#begin()
" 导航栏
Plug 'preservim/nerdtree'
Plug 'Xuyuanp/nerdtree-git-plugin' " 可以在导航目录中看到 git 版本信息

" 括号等标签自动补全
Plug 'jiangmiao/auto-pairs'

" themes
Plug 'crusoexia/vim-monokai'

Plug 'vim-airline/vim-airline'

" lsp 相关
Plug 'prabirshrestha/vim-lsp'
Plug 'mattn/vim-lsp-settings'
Plug 'prabirshrestha/asyncomplete.vim'
Plug 'prabirshrestha/asyncomplete-lsp.vim'

" 模糊搜索
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'

Plug 'ryanoasis/vim-devicons'
call plug#end()

/*
    Base setup
    Ref: https://www.ruanyifeng.com/blog/2018/09/vimrc.html  
*/
set nocompatible
syntax on
set showmode
set showcmd
set mouse=a
set encoding=utf-8
filetype indent on
set autoindent
set tabstop=2
set shiftwidth=2
set expandtab
set softtabstop=2
set hlsearch
set ignorecase
set nu
set cursorline
set nowrap
set termguicolors " set t_Co=256
set backspace=indent,eol,start
colorscheme monokai

" keymap
nmap o o<esc>k                      " normal 模式下按 o 当前行下方增加空行
nmap ss <cmd>wa<CR>                 " normal 模式下按 ss 将所有 buffer 中的内容写入对应文件
nmap nh <cmd>nohl<CR>               " 通常用于搜索指定内容后，取消匹配项的高亮
nmap nt <cmd>NERDTreeToggle<CR>     " 开启导航栏
nmap <C-a> gg<S-v><S-g>             " ctrl+a 全选
nmap tf <cmd>Files<CR>              " 模糊搜索文件名称
nmap tl <cmd>RG<CR>                 " 模糊搜索文件内容

" autocmd
autocmd WinLeave * setlocal nocursorline
autocmd WinEnter * setlocal cursorline

/*
    Setup lsp
*/
function!  s:on_lsp_buffer_enabled() abort
    setlocal omnifunc=lsp#complete
    setlocal signcolumn=yes
    if exists('+tagfunc') | setlocal tagfunc=lsp#tagfunc | endif
    nmap <buffer> gd <plug>(lsp-definition)
    nmap <buffer> gs <plug>(lsp-document-symbol-search)
    nmap <buffer> gS <plug>(lsp-workspace-symbol-search)
    nmap <buffer> gr <plug>(lsp-references)
    nmap <buffer> gi <plug>(lsp-implementation)
    nmap <buffer> gt <plug>(lsp-type-definition)
    nmap <buffer> <leader>rn <plug>(lsp-rename)
    nmap <buffer> [d <plug>(lsp-previous-diagnostic)
    nmap <buffer> ]d <plug>(lsp-next-diagnostic)
    nmap <buffer> gh <plug>(lsp-hover)
    nnoremap <buffer> <expr><c-f> lsp#scroll(+4)
    nnoremap <buffer> <expr><c-d> lsp#scroll(-4)

    let g:lsp_format_sync_timeout = 1000
    autocmd! BufWritePre *.* call execute('LspDocumentFormatSync')
endfunction

augroup lsp_install
    au!
    " call s:on_lsp_buffer_enabled only for languages that has the server registered.
    autocmd User lsp_buffer_enabled call s:on_lsp_buffer_enabled()
augroup END

/*
    这里记不清具体作用了
*/
inoremap <expr> <Tab>   pumvisible() ? "\<C-n>" : "\<Tab>"
inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"
inoremap <expr> <cr>    pumvisible() ? asyncomplete#close_popup() : "\<cr>"
```

#### Neovim

- [通过 AppImage 安装](https://github.com/neovim/neovim/blob/master/INSTALL.md#appimage-universal-linux-package)
- [Neovim 配置](https://github.com/xdsdmg/neovim-config)

``` bash
# 导入 Neovim 配置
git clone https://github.com/xdsdmg/neovim-config.git --depth 1 $HOME/.config/nvim
```

#### tmux

``` bash
# 安装 tmux
sudo apt install tmux
```

[tmux 配置](https://github.com/xdsdmg/tmux-config)

``` bash
# 导入 tmux 配置
git clone https://github.com/xdsdmg/tmux-config.git --depth 1 $HOME/.config/tmux
```

### WSL 代理配置

``` bash
# Set proxy
is_proxy_open=0

function pp {
  if [ $is_proxy_open -eq 0 ]; then
    host_ip=127.0.0.1 # 新版本的 WSL 已支持与宿主机共享 ip
    port=7890
    url="http://${host_ip}:${port}"

    export http_proxy=$url
    export https_proxy=$url
    
    is_proxy_open=1

    echo "proxy is open, proxy url: $url"

    return
  fi

  is_proxy_open=0
  export http_proxy=
  export https_proxy=
  echo "proxy is closed"
}
```

### Windows 快捷键适配 Mac

[Windows 快捷键适配 Mac](https://juejin.cn/post/7162921939198017567)
