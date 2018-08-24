// Copyright (c) 2016 The Decred developers.

// +build cuda,!opencl

package main

/*
#cgo CXXFLAGS: -O3 -march=x86-64 -mtune=generic -std=c++17 -Wall -Wno-strict-aliasing -Wno-shift-count-overflow -Werror
#cgo !windows LDFLAGS: -z muldefs -L/opt/cuda/lib64 -L/opt/cuda/lib -L/usr/local/cuda/lib64 -leqcuda1445 -lcuda -lcudart -lstdc++ -ldl
// TODO:               -z muldefs   is workaround and requires CGO_LDFLAGS_ALLOW='.*' env variable
#cgo windows LDFLAGS: -Lobj -ldecred -Lnvidia/CUDA/v7.0/lib/x64 -lcuda -lcudart -Lnvidia/NVSMI -lnvml
*/
import "C"
