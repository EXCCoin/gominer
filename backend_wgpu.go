//go:build wgpu

package main

// wgpu backend: portable solver (Vulkan/Metal/DX12 — NVIDIA, AMD, Intel,
// Apple) from the eqwgpu1445 Rust crate, which implements the same
// eq_create/eq_solve/eq_destroy ABI as the CUDA library.

/*
#cgo LDFLAGS: -L. -leqwgpu1445 -ldl -lpthread -lm
#include <stdint.h>
extern uint32_t eq_adapter_count();
extern int eq_adapter_name(uint32_t index, char *buf, uint32_t len);
extern void eq_set_adapter(uint32_t index);
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func backendName() string { return "wgpu" }

// backendEnumerate returns the names of usable GPUs, index == adapter ordinal.
func backendEnumerate() ([]string, error) {
	count := int(C.eq_adapter_count())
	if count < 1 {
		return nil, fmt.Errorf("no wgpu adapters found")
	}
	names := make([]string, count)
	for i := 0; i < count; i++ {
		buf := make([]byte, 256)
		if C.eq_adapter_name(C.uint32_t(i), (*C.char)(unsafe.Pointer(&buf[0])), 256) != 0 {
			names[i] = "unknown adapter"
			continue
		}
		n := 0
		for n < len(buf) && buf[n] != 0 {
			n++
		}
		names[i] = string(buf[:n])
	}
	return names, nil
}

// backendBindThread selects the adapter that eq_create on this thread uses.
func backendBindThread(ordinal int) {
	C.eq_set_adapter(C.uint32_t(ordinal))
}

func backendAutoInstances(ordinal int) int {
	// One instance is the safe portable default (~2.2GB). Raise it only after
	// benchmarking the specific adapter with --instances.
	return 1
}

func backendRelease(ordinal int) error { return nil }

// backendDeviceStats returns (fan percent, temperature C); not available
// through wgpu, so devices simply show no fan/temp telemetry.
func backendDeviceStats(index int) (uint32, uint32) {
	return 0, 0
}
