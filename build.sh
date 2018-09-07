#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd "$DIR" > /dev/null

mkdir -p obj
g++ -O3 -march=x86-64 -mtune=generic -fPIC -c eqcuda1445/blake/blake2b.cpp -o obj/blake.o
nvcc -arch sm_35 -O3 -Xptxas -O3 -Xcompiler -O3 --compiler-options '-fPIC' -rdc=true -c -o obj/solver.o eqcuda1445/solver.cu
nvcc -arch sm_35 -O3 -Xptxas -O3 -Xcompiler -O3 --compiler-options '-fPIC' -dlink -o obj/eqcuda1445.o obj/solver.o
g++ -O3 -march=x86-64 -mtune=generic -fPIC -Wl,-soname,libeqcuda1445.so -shared -o libeqcuda1445.so obj/eqcuda1445.o obj/solver.o obj/blake.o -lcudart_static -ldl -lrt -lpthread

dep ensure
go build
sudo cp libeqcuda1445.so /usr/lib
