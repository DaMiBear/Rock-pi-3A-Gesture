#!/bin/bash
cd build
rm * -rf
cmake -DCMAKE_TOOLCHAIN_FILE="../aarch64-linux-gnu.toolchain.cmake" ..
make -j$(nproc)
cd ..