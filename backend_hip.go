//go:build hip

package main

// HIP backend: AMD device management and the native Equihash solver.

/*
#cgo LDFLAGS: -L. -leqhip1445 -lamdhip64 -lstdc++
#include <hip/hip_runtime_api.h>
*/
import "C"

import (
	"fmt"
	"runtime"
)

func backendName() string { return "HIP" }

func hipError(operation string, status C.hipError_t) error {
	return fmt.Errorf("%s: %s", operation, C.GoString(C.hipGetErrorString(status)))
}

func backendEnumerate() ([]string, error) {
	if status := C.hipInit(0); status != C.hipSuccess {
		return nil, hipError("HIP initialization failed", status)
	}

	var count C.int
	if status := C.hipGetDeviceCount(&count); status != C.hipSuccess {
		return nil, hipError("HIP device enumeration failed", status)
	}
	if count < 1 {
		return nil, fmt.Errorf("no HIP devices found")
	}

	names := make([]string, int(count))
	for i := range names {
		if status := C.hipSetDevice(C.int(i)); status != C.hipSuccess {
			return nil, hipError(fmt.Sprintf("HIP device %d selection failed", i), status)
		}
		if status := C.hipSetDeviceFlags(C.uint(C.hipDeviceScheduleYield)); status != C.hipSuccess {
			return nil, hipError(fmt.Sprintf("HIP device %d scheduling setup failed", i), status)
		}
		var properties C.hipDeviceProp_t
		if status := C.hipGetDeviceProperties(&properties, C.int(i)); status != C.hipSuccess {
			return nil, hipError(fmt.Sprintf("HIP device %d properties failed", i), status)
		}
		names[i] = C.GoString(&properties.name[0])
	}
	return names, nil
}

func backendBindThread(ordinal int) {
	if status := C.hipSetDevice(C.int(ordinal)); status != C.hipSuccess {
		panic(hipError(fmt.Sprintf("HIP device %d selection failed", ordinal), status))
	}
}

// One solver already saturates this APU and holds roughly 3 GB of buffers.
func backendAutoInstances(int) int { return 1 }

func backendRelease(ordinal int) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if status := C.hipSetDevice(C.int(ordinal)); status != C.hipSuccess {
		return hipError(fmt.Sprintf("HIP device %d selection failed", ordinal), status)
	}
	if status := C.hipDeviceReset(); status != C.hipSuccess {
		return hipError("HIP device reset failed", status)
	}
	return nil
}

func backendDeviceStats(int) (uint32, uint32) { return 0, 0 }
