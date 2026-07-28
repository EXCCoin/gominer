//go:build !wgpu

package main

// CUDA backend: device management via the cu wrappers, temperature/fan
// telemetry via NVML, solver from libeqcuda1445.a.

/*
#cgo LDFLAGS: -L. -leqcuda1445 -lstdc++
*/
import "C"
import (
	"fmt"
	"runtime"
	"sync"

	"github.com/EXCCoin/gominer/cu"
	"github.com/EXCCoin/gominer/nvml"
)

var (
	nvmlInitOnce sync.Once
	nvmlInitErr  error
)

func backendName() string { return "CUDA" }

// backendEnumerate returns the names of usable GPUs, index == device ordinal.
func backendEnumerate() ([]string, error) {
	cu.Init(0)

	version := cu.Version()
	maj := version / 1000
	min := version % 100
	if maj < 5 || (maj == 5 && min < 5) {
		return nil, fmt.Errorf("driver does not support CUDA 5.5 API")
	}

	count := cu.DeviceGetCount()
	if count < 1 {
		return nil, fmt.Errorf("no CUDA devices found")
	}
	names := make([]string, count)
	for i := 0; i < count; i++ {
		names[i] = cu.DeviceGet(i).Name()
	}
	return names, nil
}

// backendBindThread prepares the calling (OS-locked) thread to run solver
// instances on the given device.
func backendBindThread(ordinal int) {
	cu.SetDevice(cu.Device(ordinal))
}

// Concurrent instances hide each other's launch/readback bubbles; each holds
// ~2.7 GB of buckets. Size the default to VRAM, capped where the GPU is
// saturated anyway; -I overrides.
func backendAutoInstances(ordinal int) (n int) {
	n = 1
	defer func() { _ = recover() }() // cu wrappers panic; keep the fallback
	cu.SetDevice(cu.Device(ordinal))
	_, total := cu.MemGetInfo()
	spare := int64(total) - 2<<30 // headroom for context + display
	if v := int(spare / (2800 << 20)); v > n {
		n = v
	}
	if n > 4 {
		n = 4
	}
	return n
}

func backendRelease(ordinal int) (err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	defer func() {
		if recovered := recover(); recovered != nil {
			err = fmt.Errorf("CUDA device reset failed: %v", recovered)
		}
	}()

	cu.SetDevice(cu.Device(ordinal))
	cu.DeviceReset()
	return nil
}

// backendDeviceStats returns (fan percent, temperature C), zeros if unknown.
func backendDeviceStats(index int) (uint32, uint32) {
	nvmlInitOnce.Do(func() { nvmlInitErr = nvml.Init() })
	if nvmlInitErr != nil {
		minrLog.Debugf("NVML Init error: %v", nvmlInitErr)
		return 0, 0
	}

	busID := cu.DevicePCIBusID(index)
	dh, err := nvml.DeviceGetHandleByPCIBusID(busID)
	if err != nil {
		minrLog.Debugf("NVML DeviceGetHandleByPCIBusID(%s) error: %v", busID, err)
		return 0, 0
	}

	fanPercent, temperature := uint32(0), uint32(0)
	if speed, err := nvml.DeviceFanSpeed(dh); err == nil {
		fanPercent = uint32(speed)
	}
	if temp, err := nvml.DeviceTemperature(dh); err == nil {
		temperature = uint32(temp)
	}
	return fanPercent, temperature
}
