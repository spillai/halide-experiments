Halide installation
====

Follow instructions on https://github.com/halide/Halide<br>
Build straight from source, with terminfo off, otherwise python import
of halide will complain

```sh
git clone https://github.com/halide/Halide.git
mkdir halide_build
cd halide_build
svn co https://llvm.org/svn/llvm-project/llvm/branches/release_37 llvm3.7
svn co https://llvm.org/svn/llvm-project/cfe/branches/release_37 llvm3.7/tools/clang

cd llvm3.7
mkdir build
cd build
cmake -DLLVM_ENABLE_TERMINFO=OFF
-DLLVM_TARGETS_TO_BUILD="X86;ARM;NVPTX;AArch64;Mips;PowerPC"
-DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j8

# wget http://llvm.org/releases/3.7.1/clang+llvm-3.7.1-x86_64-linux-gnu-ubuntu-14.04.tar.xz
# tar -xvf clang+llvm-3.7.1-x86_64-linux-gnu-ubuntu-14.04.tar.xz
# cd clang+llvm-3.7.1-x86_64-linux-gnu-ubuntu-14.04

export LLVM_CONFIG=`pwd`/bin/llvm-config
export CLANG=`pwd`/bin/clang
```

# Make halide with Makefile
```sh
cd halide_build/
mkdir halide && cd halide
make -f ../../Halide/Makefile -j 20
export HALIDE_ROOT_DIR=`pwd`
```
# Make halide with CMake
```sh
mkdir cmake_build
cd cmake_build
export LLVM_ROOT=/path/to/llvm3.7/build
cmake -DLLVM_BIN=${LLVM_ROOT}/bin -DLLVM_INCLUDE="${LLVM_ROOT}/../include;${LLVM_ROOT}/include" -DLLVM_LIB=${LLVM_ROOT}/lib -DLLVM_VERSION=37 ..
make -j8
```

# Setup env for python binding
```sh
cd Halide/python_bindings
mkdir build && cd build
cmake ../ -DUSE_PYTHON=2 -DHALIDE_ROOT_DIR=$HALIDE_ROOT_DIR -DUSE_BOOST_NUMPY=OFF
make -j 
```
