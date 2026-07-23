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

	"github.com/EXCCoin/gominer/cu"
	"github.com/EXCCoin/gominer/nvml"
)

var nvmlInitialized = false

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

// backendAutoInstances sizes the number of solver instances to free VRAM.
func backendAutoInstances(ordinal int) int {
	cu.SetDevice(cu.Device(ordinal))
	free, _ := cu.MemGetInfo()
	n := int((free - (1500 << 20)) / solverMemBytes)
	if n < 1 {
		n = 1
	}
	if n > 8 {
		n = 8
	}
	return n
}

func backendRelease(ordinal int) {
	cu.SetDevice(cu.Device(ordinal))
	cu.DeviceReset()
}

// backendDeviceStats returns (fan percent, temperature C), zeros if unknown.
func backendDeviceStats(index int) (uint32, uint32) {
	if !nvmlInitialized {
		if err := nvml.Init(); err != nil {
			minrLog.Debugf("NVML Init error: %v", err)
			return 0, 0
		}
		nvmlInitialized = true
	}

	dh, err := nvml.DeviceGetHandleByIndex(index)
	if err != nil {
		minrLog.Debugf("NVML DeviceGetHandleByIndex error: %v", err)
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
