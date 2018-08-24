// Copyright (c) 2016 The Decred developers.

// +build cuda,!opencl

package main

/*
#cgo !windows LDFLAGS: -L/opt/cuda/lib64 -L/opt/cuda/lib -L/usr/local/cuda/lib64 obj/decred.a -lcuda -lcudart -lstdc++ -ldl
#cgo windows LDFLAGS: -Lobj -ldecred -Lnvidia/CUDA/v7.0/lib/x64 -lcuda -lcudart -Lnvidia/NVSMI -lnvml
*/
import "C"
