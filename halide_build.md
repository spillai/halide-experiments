Halide installation
====

git clone https://github.com/halide/Halide.git
mkdir halide_build
cd halide_build
wget http://llvm.org/releases/3.7.1/clang+llvm-3.7.1-x86_64-linux-gnu-ubuntu-14.04.tar.xz
tar -xvf clang+llvm-3.7.1-x86_64-linux-gnu-ubuntu-14.04.tar.xz
cd clang+llvm-3.7.1-x86_64-linux-gnu-ubuntu-14.04

export LLVM_CONFIG=`pwd`/bin/llvm-config
export CLANG=`pwd`/bin/clang


# Make halide with Makefile
cd halide_build/
mkdir halide && cd halide
make -f ../../Halide/Makefile -j 20

# Make halide with CMake
mkdir cmake_build
cd cmake_build
export LLVM_ROOT=/path/to/llvm3.7/build
cmake -DLLVM_BIN=${LLVM_ROOT}/bin -DLLVM_INCLUDE="${LLVM_ROOT}/../include;${LLVM_ROOT}/include" -DLLVM_LIB=${LLVM_ROOT}/lib -DLLVM_VERSION=37 ..
make -j8
