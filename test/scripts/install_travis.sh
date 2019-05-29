#!/bin/bash -x

# Download the right llvm release
wget -q https://releases.llvm.org/${LLVM_VERSION}/clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-14.04.tar.xz
tar xf clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-14.04.tar.xz
mv clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-14.04 ${TRAVIS_BUILD_DIR}/llvm