UNIX_CXX= g++ -O3 -march=x86-64 -mtune=generic -std=c++17 -fPIC
UNIX_NVCC= nvcc -arch sm_35 -O3 -Xptxas -O3 -Xcompiler -O3 --compiler-options '-fPIC'

WIN_CXX= g++ -O3 -march=x86-64 -mtune=generic -std=c++17 -fPIC
WIN_NVCC= nvcc -arch sm_35 -O3 -Xptxas -O3 -Xcompiler -O3 --compiler-options '-fPIC'

.DEFAULT_GOAL := build

ifeq ($(OS),Windows_NT)
obj:
else
obj:
endif
	mkdir obj

ifeq ($(OS),Windows_NT)
obj/blake.o: obj eqcuda1445/blake/blake2b.cpp
	$(WIN_CXX) -c eqcuda1445/blake/blake2b.cpp -o obj/blake.o

obj/solver.o: obj eqcuda1445/solver.cu
	$(WIN_NVCC) -rdc=true -c -o obj/solver.o eqcuda1445/solver.cu

obj/eqcuda1445.o: obj obj/solver.o
	$(WIN_NVCC) -dlink -o obj/eqcuda1445.o obj/solver.o

obj/libeqcuda1445.dll: obj obj/blake.o obj/eqcuda1445.o
	$(WIN_CXX) -Wl,-soname,libeqcuda1445.so -shared -o obj/libeqcuda1445.dll obj/eqcuda1445.o obj/solver.o obj/blake.o -lcudart_static  -ldl -lrt -lpthread

else

obj/blake.o: obj eqcuda1445/blake/blake2b.cpp
	$(UNIX_CXX) -c eqcuda1445/blake/blake2b.cpp -o obj/blake.o

obj/solver.o: obj eqcuda1445/solver.cu
	$(UNIX_NVCC) -rdc=true -c -o obj/solver.o eqcuda1445/solver.cu

obj/eqcuda1445.o: obj obj/solver.o
	$(UNIX_NVCC) -dlink -o obj/eqcuda1445.o obj/solver.o

obj/libeqcuda1445.so: obj obj/blake.o obj/eqcuda1445.o
	$(UNIX_CXX) -Wl,-soname,libeqcuda1445.so -shared -o obj/libeqcuda1445.so obj/eqcuda1445.o obj/solver.o obj/blake.o -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lcudart_static  -ldl -lrt -lpthread
endif


ifeq ($(OS),Windows_NT)
build: obj/libeqcuda1445.dll
else
build: obj/libeqcuda1445.so
endif
	go build

ifeq ($(OS),Windows_NT)
install: obj/libeqcuda1445.dll
else
install: obj/libeqcuda1445.so
endif
	go install

clean:
	rm -rf obj
	go clean
