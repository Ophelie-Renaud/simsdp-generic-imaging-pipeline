#!/bin/sh
rm -r bin
mkdir bin
cd bin

# Generating the Makefile
# Run ccmake gui to debug cmake problems
cmake ../GPU -DCMAKE_BUILD_TYPE=Release
make clean
make all

./SEP_Pipeline_GPU
