CXX= g++ -O3 -march=x86-64 -mtune=generic -std=c++17 -fPIC
NVCC ?= nvcc -arch sm_35 -O3 -Xptxas -O3 -Xcompiler -O3 --compiler-options '-fPIC'
AR ?= ar
# -o is gnu only so this needs to be smarter; it does work because on darwin it
#  fails which is also not windows.
ARCH:=$(shell uname -o)

.DEFAULT_GOAL := build

ifeq ($(ARCH),Msys)
nvidia:
endif

# Windows needs additional setup and since cgo does not support spaces in
# in include and library paths we copy it to the correct location.
#
# Windows build assumes that CUDA V7.0 is installed in its default location.
#
# Windows gominer requires nvml.dll and decred.dll to reside in the same
# directory as gominer.exe.
ifeq ($(ARCH),Msys)
obj: nvidia
	mkdir nvidia
	cp -r /c/Program\ Files/NVIDIA\ GPU\ Computing\ Toolkit/* nvidia
	cp -r /c/Program\ Files/NVIDIA\ Corporation/NVSMI nvidia
else
obj:
endif
	mkdir obj



ifeq ($(ARCH),Msys)
obj/eqcuda1445.so: # TODO
	$(NVCC) # TODO
else
obj/blake.o: obj eqcuda1445/blake/blake2b.cpp
	$(CXX) -c eqcuda1445/blake/blake2b.cpp -o obj/blake.o

obj/solver.o: obj eqcuda1445/solver.cu
	$(NVCC) -rdc=true -c -o obj/solver.o eqcuda1445/solver.cu

obj/eqcuda1445.o: obj obj/solver.o
	$(NVCC) -dlink -o obj/eqcuda1445.o obj/solver.o

obj/libeqcuda1445.so: obj obj/eqcuda1445.o obj/blake.o
	$(CXX) -Wl,-soname,libeqcuda1445.so -shared -o obj/libeqcuda1445.so obj/eqcuda1445.o obj/solver.o obj/blake.o -L/opt/cuda/lib64 -L/usr/local/cuda/lib64 -lcudart_static  -ldl -lrt -lpthread
endif


ifeq ($(ARCH),Msys)
build: # TODO
else
build: obj/libeqcuda1445.so
endif
	go build

ifeq ($(ARCH),Msys)
install: # TODO
else
install: obj/libeqcuda1445.so
endif
	go install

clean:
	rm -rf obj
	go clean
ifeq ($(ARCH),Msys)
	rm -rf nvidia
endif
