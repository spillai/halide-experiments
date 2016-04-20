export CLANG_BINARY_DIR=`pwd`/halide_build/llvm3.7/build
export LLVM_CONFIG=$CLANG_BINARY_DIR/bin/llvm-config
export CLANG=$CLANG_BINARY_DIR/bin/clang

export HALIDE_ROOT_DIR=`pwd`/halide_build/build
export HALIDE_PYTHON_DIR=`pwd`/Halide/python_bindings/build

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HALIDE_ROOT_DIR/lib
export PYTHONPATH=$PYTHONPATH:$HALIDE_ROOT_DIR/lib:$HALIDE_PYTHON_DIR
