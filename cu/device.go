//go:build !wgpu && !hip

package cu

// This file implements the minimal CUDA device management gominer needs.

/*
#cgo CXXFLAGS: -O3 -Wno-deprecated-declarations
#cgo CFLAGS: -O3 -Wno-deprecated-declarations
#cgo !windows CFLAGS: -I/usr/local/cuda/include
#cgo !windows LDFLAGS: -lcuda -lcudart_static -ldl -lrt -lpthread
#cgo windows LDFLAGS: -lcuda -lcudart
#include <cuda.h>
#include <cuda_runtime.h>
*/
import "C"

// CUDA Device number.
type Device int

// Initialize the CUDA driver API.
func Init(flags int) {
	err := Result(C.cuInit(C.uint(flags)))
	if err != SUCCESS {
		panic(err)
	}
}

// Returns the CUDA driver version.
func Version() int {
	var version C.int
	err := Result(C.cuDriverGetVersion(&version))
	if err != SUCCESS {
		panic(err)
	}
	return int(version)
}

// Returns the number of CUDA-capable devices.
func DeviceGetCount() int {
	var count C.int
	err := Result(C.cuDeviceGetCount(&count))
	if err != SUCCESS {
		panic(err)
	}
	return int(count)
}

// Returns a device handle given an ordinal in the range [0, DeviceGetCount()-1].
func DeviceGet(ordinal int) Device {
	var device C.CUdevice
	err := Result(C.cuDeviceGet(&device, C.int(ordinal)))
	if err != SUCCESS {
		panic(err)
	}
	return Device(device)
}

// Gets the name of the device.
func (dev Device) Name() string {
	buf := make([]C.char, 256)
	err := Result(C.cuDeviceGetName(&buf[0], 256, C.CUdevice(dev)))
	if err != SUCCESS {
		panic(err)
	}
	return C.GoString(&buf[0])
}

// DevicePCIBusID returns the PCI identity for a CUDA device ordinal.
func DevicePCIBusID(ordinal int) string {
	buf := make([]C.char, 32)
	err := Result(C.cudaDeviceGetPCIBusId(&buf[0], C.int(len(buf)), C.int(ordinal)))
	if err != SUCCESS {
		panic(err)
	}
	return C.GoString(&buf[0])
}

// Set the device as current for the calling thread.
func SetDevice(device Device) {
	err := Result(C.cudaSetDevice(C.int(device)))
	if err != SUCCESS {
		panic(err)
	}
}

// Reset the state of the current device.
func DeviceReset() {
	err := Result(C.cudaDeviceReset())
	if err != SUCCESS {
		panic(err)
	}
}

// MemGetInfo returns the free and total memory in bytes of the current device.
func MemGetInfo() (free, total uint64) {
	var f, t C.size_t
	err := Result(C.cudaMemGetInfo(&f, &t))
	if err != SUCCESS {
		panic(err)
	}
	return uint64(f), uint64(t)
}
