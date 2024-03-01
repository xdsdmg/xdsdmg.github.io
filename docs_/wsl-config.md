# Windows 与 WSL 配置

## Vim 配置

``` vim
call plug#begin()
Plug 'preservim/nerdtree'
Plug 'Xuyuanp/nerdtree-git-plugin' " 可以在导航目录中看到 git 版本信息

Plug 'jiangmiao/auto-pairs'

" themes
Plug 'crusoexia/vim-monokai'

Plug 'vim-airline/vim-airline'

Plug 'prabirshrestha/vim-lsp'
Plug 'mattn/vim-lsp-settings'
Plug 'prabirshrestha/asyncomplete.vim'
Plug 'prabirshrestha/asyncomplete-lsp.vim'
Plug 'junegunn/fzf', { 'do': { -> fzf#install() } }
Plug 'junegunn/fzf.vim'
Plug 'ryanoasis/vim-devicons'
call plug#end()

" base setup
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
nmap o o<esc>k
nmap ss <cmd>wa<CR> 
nmap nh <cmd>nohl<CR>
nmap nt <cmd>NERDTreeToggle<CR>
nmap <C-a> gg<S-v><S-g>
nmap qq <cmd>%s/$/\\n\\n/g<CR>gg1000<S-j>
nmap tf <cmd>Files<CR>
nmap tl <cmd>RG<CR>

" autocmd
autocmd WinLeave * setlocal nocursorline
autocmd WinEnter * setlocal cursorline

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

inoremap <expr> <Tab>   pumvisible() ? "\<C-n>" : "\<Tab>"
inoremap <expr> <S-Tab> pumvisible() ? "\<C-p>" : "\<S-Tab>"
inoremap <expr> <cr>    pumvisible() ? asyncomplete#close_popup() : "\<cr>"
```

## WSL 代理配置

``` bash
# Set proxy
is_proxy_open=0

function pp {
  if [ $is_proxy_open -eq 0 ]; then
    host_ip=$(cat /etc/resolv.conf | grep nameserver | awk '{ print $2 }')
    url="http://${host_ip}:7890"

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

## Windows 快捷键适配 Mac

[Windows 快捷键适配 Mac](https://juejin.cn/post/7162921939198017567)
