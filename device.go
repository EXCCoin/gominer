// Copyright (c) 2016 The Decred developers.

package main

/*
#cgo CXXFLAGS: -O3 -march=x86-64 -mtune=generic -std=c++17 -Wall -Wno-strict-aliasing -Wno-shift-count-overflow -Werror
#cgo !windows LDFLAGS: -Lobj -leqcuda1445
#cgo windows LDFLAGS: -Lobj -leqcuda1445
#include "eqcuda1445/eqcuda1445.h"
*/
import "C"
import (
	"bytes"
	"encoding/binary"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/EXCCoin/exccd/blockchain"
	"github.com/EXCCoin/exccd/chaincfg"
	"github.com/EXCCoin/exccd/wire"

	"github.com/EXCCoin/gominer/nvml"
	"github.com/EXCCoin/gominer/util"
	"github.com/EXCCoin/gominer/work"

	"github.com/barnex/cuda5/cu"
	cptr "github.com/mattn/go-pointer"
)

//export equihashProxyGominer
func equihashProxyGominer(userData unsafe.Pointer, solution unsafe.Pointer) C.int {
	device := cptr.Restore(userData).(*Device)
	csol := C.GoBytes(solution, C.int(equihashSolutionSize(chaincfg.MainNetParams.N, chaincfg.MainNetParams.K)))
	device.handleEquihashSolution(csol)
	return 0
}

var deviceLibraryInitialized = false

// Constants for fan and temperature bits
const (
	ChangeLevelNone           = "None"
	ChangeLevelSmall          = "Small"
	ChangeLevelLarge          = "Large"
	DeviceKindAMDGPU          = "AMDGPU"
	DeviceKindADL             = "ADL"
	DeviceKindNVML            = "NVML"
	DeviceTypeGPU             = "GPU"
	FanControlHysteresis      = uint32(3)
	FanControlAdjustmentLarge = uint32(10)
	FanControlAdjustmentSmall = uint32(5)
	SeverityLow               = "Low"
	SeverityHigh              = "High"
	TargetLower               = "Lower"
	TargetHigher              = "Raise"
	TargetNone                = "None"
)

type Device struct {
	// The following variables must only be used atomically.
	fanPercent  uint32
	temperature uint32

	sync.Mutex
	index int
	cuda  bool

	deviceName               string
	deviceType               string
	fanTempActive            bool
	fanControlActive         bool
	fanControlLastTemp       uint32
	fanControlLastFanPercent uint32
	kind                     string
	tempTarget               uint32

	// Items for CUDA device
	cuDeviceID     cu.Device
	cuInSize       int64
	cuOutputBuffer []float64

	workSize uint32

	// extraNonce is the device extraNonce, where the first
	// byte is the device ID (supporting up to 255 devices)
	// while the last 3 bytes is the extraNonce value. If
	// the extraNonce goes through all 0x??FFFFFF values,
	// it will reset to 0x??000000.
	extraNonce    uint32
	currentWorkID uint32

	midstate  [8]uint32
	lastBlock [16]uint32

	work     work.Work
	newWork  chan *work.Work
	workDone chan []byte
	hasWork  bool

	started          uint32
	allDiffOneShares uint64
	validShares      uint64
	invalidShares    uint64

	quit chan struct{}
}

func (d *Device) Run() {
	err := d.runDevice()
	if err != nil {
		minrLog.Errorf("Error on device: %v", err)
	}
}

func (d *Device) Stop() {
	close(d.quit)
}

func (d *Device) SetWork(w *work.Work) {
	d.newWork <- w
}

func (d *Device) PrintStats() {
	secondsElapsed := uint32(time.Now().Unix()) - d.started
	if secondsElapsed == 0 {
		return
	}

	d.Lock()
	defer d.Unlock()

	averageHashRate, fanPercent, temperature := d.Status()
	log := fmt.Sprintf("DEV #%d (%s) %v", d.index, d.deviceName, util.FormatHashRate(averageHashRate))

	if fanPercent != 0 {
		log = fmt.Sprintf("%s Fan=%v%%", log, fanPercent)
	}

	if temperature != 0 {
		log = fmt.Sprintf("%s T=%vC", log, temperature)
	}

	minrLog.Info(log)
}

// UpdateFanTemp updates a device's statistics
func (d *Device) UpdateFanTemp() {
	d.Lock()
	defer d.Unlock()
	if d.fanTempActive {
		// For now amd and nvidia do more or less the same thing
		// but could be split up later.  Anything else (Intel) just
		// doesn't do anything.
		switch d.kind {
		case DeviceKindADL, DeviceKindAMDGPU, DeviceKindNVML:
			fanPercent, temperature := deviceStats(d.index)
			atomic.StoreUint32(&d.fanPercent, fanPercent)
			atomic.StoreUint32(&d.temperature, temperature)
			break
		}
	}
}

func (d *Device) Status() (float64, uint32, uint32) {
	secondsElapsed := uint32(time.Now().Unix()) - d.started

	averageHashRate := float64(d.allDiffOneShares) / float64(secondsElapsed)

	fanPercent := atomic.LoadUint32(&d.fanPercent)
	temperature := atomic.LoadUint32(&d.temperature)

	return averageHashRate, fanPercent, temperature
}

func (d *Device) Release() {
	cu.SetDevice(d.cuDeviceID)
	cu.DeviceReset()
}

func (d *Device) handleEquihashSolution(solution []byte) {
	minrLog.Debugf("GPU #%d: Found candidate: %08x, workID %08x, timestamp %08x",
		d.index, solution, util.Uint32EndiannessSwap(d.currentWorkID), d.lastBlock[work.TimestampWord])

	// Assess the work. If it's below target, it'll be rejected
	// here. The mining algorithm currently sends this function any
	// difficulty 1 shares.
	d.foundCandidate(d.lastBlock[work.TimestampWord], solution)
}

func (d *Device) updateCurrentWork() {
	var w *work.Work
	if d.hasWork {
		// If we already have work, we just need to check if there's new one
		// without blocking if there's not.
		select {
		case w = <-d.newWork:
		default:
			return
		}
	} else {
		// If we don't have work, we block until we do. We need to watch for
		// quit events too.
		select {
		case w = <-d.newWork:
		case <-d.quit:
			return
		}
	}

	d.work = *w

	// Bump and set the work ID if the work is new.
	d.currentWorkID++
	d.hasWork = true
}

// This is pretty hacky/proof-of-concepty
func (d *Device) fanControl() {
	d.Lock()
	defer d.Unlock()
	var fanChangeLevel, fanIntent string
	var fanChange uint32
	fanLast := d.fanControlLastFanPercent

	var tempChange uint32
	var tempChangeLevel, tempDirection string
	var tempSeverity, tempTargetType string

	var firstRun bool

	tempLast := d.fanControlLastTemp
	tempMinAllowed := d.tempTarget - FanControlHysteresis
	tempMaxAllowed := d.tempTarget + FanControlHysteresis

	// Save the values we read for the next time the loop is run
	fanCur := atomic.LoadUint32(&d.fanPercent)
	tempCur := atomic.LoadUint32(&d.temperature)
	d.fanControlLastFanPercent = fanCur
	d.fanControlLastTemp = tempCur

	// if this is our first run then set some more variables
	if tempLast == 0 && fanLast == 0 {
		fanLast = fanCur
		tempLast = tempCur
		firstRun = true
	}

	// Everything is OK so just return without adjustment
	if tempCur <= tempMaxAllowed && tempCur >= tempMinAllowed {
		minrLog.Tracef("DEV #%d within acceptable limits "+
			"curTemp %v is above minimum %v and below maximum %v",
			d.index, tempCur, tempMinAllowed, tempMaxAllowed)
		return
	}

	// Lower the temperature of the device
	if tempCur > tempMaxAllowed {
		tempTargetType = TargetLower
		if tempCur-tempMaxAllowed > FanControlHysteresis {
			tempSeverity = SeverityHigh
		} else {
			tempSeverity = SeverityLow
		}
	}

	// Raise the temperature of the device
	if tempCur < tempMinAllowed {
		tempTargetType = TargetHigher
		if tempMaxAllowed-tempCur >= FanControlHysteresis {
			tempSeverity = SeverityHigh
		} else {
			tempSeverity = SeverityLow
		}
	}

	// we increased the fan to lower the device temperature last time
	if fanLast < fanCur {
		fanChange = fanCur - fanLast
		fanIntent = TargetHigher
	}
	// we decreased the fan to raise the device temperature last time
	if fanLast > fanCur {
		fanChange = fanLast - fanCur
		fanIntent = TargetLower
	}
	// we didn't make any changes
	if fanLast == fanCur {
		fanIntent = TargetNone
	}

	if fanChange == 0 {
		fanChangeLevel = ChangeLevelNone
	} else if fanChange == FanControlAdjustmentSmall {
		fanChangeLevel = ChangeLevelSmall
	} else if fanChange == FanControlAdjustmentLarge {
		fanChangeLevel = ChangeLevelLarge
	} else {
		// XXX Seems the AMDGPU driver may not support all values or
		// changes values underneath us
		minrLog.Tracef("DEV #%d fan changed by an unexpected value %v", d.index,
			fanChange)
		if fanChange < FanControlAdjustmentSmall {
			fanChangeLevel = ChangeLevelSmall
		} else {
			fanChangeLevel = ChangeLevelLarge
		}
	}

	if tempLast < tempCur {
		tempChange = tempCur - tempLast
		tempDirection = "Up"
	}
	if tempLast > tempCur {
		tempChange = tempLast - tempCur
		tempDirection = "Down"
	}
	if tempLast == tempCur {
		tempDirection = "Stable"
	}

	if tempChange == 0 {
		tempChangeLevel = ChangeLevelNone
	} else if tempChange > FanControlHysteresis {
		tempChangeLevel = ChangeLevelLarge
	} else {
		tempChangeLevel = ChangeLevelSmall
	}

	minrLog.Tracef("DEV #%d firstRun %v fanChange %v fanChangeLevel %v "+
		"fanIntent %v tempChange %v tempChangeLevel %v tempDirection %v "+
		" tempSeverity %v tempTargetType %v", d.index, firstRun, fanChange,
		fanChangeLevel, fanIntent, tempChange, tempChangeLevel, tempDirection,
		tempSeverity, tempTargetType)

	// We have no idea if the device is starting cold or re-starting hot
	// so only adjust the fans upwards a little bit.
	if firstRun {
		if tempTargetType == TargetLower {
			fanControlSet(d.index, fanCur, tempTargetType, ChangeLevelSmall)
			return
		}
	}

	// we didn't do anything last time so just match our change to the severity
	if fanIntent == TargetNone {
		if tempSeverity == SeverityLow {
			fanControlSet(d.index, fanCur, tempTargetType, ChangeLevelSmall)
		} else {
			fanControlSet(d.index, fanCur, tempTargetType, ChangeLevelLarge)
		}
	}

	// XXX could do some more hysteresis stuff here

	// we tried to raise or lower the temperature but it didn't work so
	// do it some more according to the severity level
	if fanIntent == tempTargetType {
		if tempSeverity == SeverityLow {
			fanControlSet(d.index, fanCur, tempTargetType, ChangeLevelSmall)
		} else {
			fanControlSet(d.index, fanCur, tempTargetType, ChangeLevelLarge)
		}
	}

	// we raised or lowered the temperature too much so just do a small
	// adjustment
	if fanIntent != tempTargetType {
		fanControlSet(d.index, fanCur, tempTargetType, ChangeLevelSmall)
	}
}

func (d *Device) fanControlSupported(kind string) bool {
	fanControlDrivers := []string{DeviceKindADL, DeviceKindAMDGPU}

	for _, driver := range fanControlDrivers {
		if driver == kind {
			return true
		}
	}
	return false
}

func (d *Device) foundCandidate(ts uint32, solution []byte) {
	d.Lock()
	defer d.Unlock()

	// Construct the final block header.
	copy(d.work.BlockHeader.EquihashSolution[:], solution)

	hashNum := d.work.BlockHeader.BlockHash()
	hashNumBig := blockchain.HashToBig(&hashNum)

	if hashNumBig.Cmp(blockchain.CompactToBig(d.work.BlockHeader.Bits)) > 0 {
		minrLog.Debugf("DEV #%d Found hash %s above minimum target %s",
			d.index, hashNumBig.String(), blockchain.CompactToBig(d.work.BlockHeader.Bits).String())
		d.invalidShares++
		return
	}

	d.allDiffOneShares++

	if !cfg.Benchmark {
		// Assess versus the pool or daemon target.
		if hashNumBig.Cmp(d.work.Target) > 0 {
			minrLog.Debugf("DEV #%d Hash %s bigger than target %032x (boo)", d.index, hashNumBig, d.work.Target.Bytes())
		} else {
			minrLog.Infof("DEV #%d Found hash with work below target! %v (yay)", d.index, hashNum)
			d.validShares++
			data := make([]byte, 0, work.GetworkDataLen)
			buf := bytes.NewBuffer(data)
			err := d.work.BlockHeader.Serialize(buf)
			if err != nil {
				errStr := fmt.Sprintf("Failed to serialize data: %v", err)
				minrLog.Errorf("Error submitting work: %v", errStr)
			} else {
				data = data[:work.GetworkDataLen]
				d.workDone <- data
			}
		}
	}
}

func (d *Device) runDevice() error {
	// Bump the extraNonce for the device it's running on
	// when you begin mining. This ensures each GPU is doing
	// different work. If the extraNonce has already been
	// set for valid work, restore that.
	d.extraNonce += uint32(d.index) << 24
	d.lastBlock[work.Nonce1Word] = util.Uint32EndiannessSwap(d.extraNonce)

	// Need to have this stuff here for a device vs thread issue.
	runtime.LockOSThread()

	cu.DeviceReset()
	cu.SetDevice(d.cuDeviceID)
	cu.SetDeviceFlags(cu.DeviceScheduleBlockingSync)

	// kernel is built with nvcc, not an api call so must be done
	// at compile time.

	deviceptr := cptr.Save(d)
	defer cptr.Unref(deviceptr)

	minrLog.Infof("Started GPU #%d: %s", d.index, d.deviceName)

	for {
		d.updateCurrentWork()

		select {
		case <-d.quit:
			return nil
		default:
		}

		// Increment extraNonce.
		util.RolloverExtraNonce(&d.extraNonce)
		d.lastBlock[work.Nonce1Word] = util.Uint32EndiannessSwap(d.extraNonce)
		binary.LittleEndian.PutUint64(d.work.BlockHeader.ExtraData[:], uint64(d.extraNonce))

		// Update the timestamp. Only solo work allows you to roll the timestamp.
		ts := d.work.JobTime
		if d.work.IsGetWork {
			diffSeconds := uint32(time.Now().Unix()) - d.work.TimeReceived
			ts = d.work.JobTime + diffSeconds
		}
		d.lastBlock[work.TimestampWord] = util.Uint32EndiannessSwap(ts)

		// Generate and set nonce
		nonce, err := wire.RandomUint64()
		if err != nil {
			minrLog.Errorf("Unexpected error while generating random nonce: %v", err)
			nonce = 0
		}

		d.work.BlockHeader.Nonce = uint32(nonce)

		// Execute the kernel and follow its execution time.
		currentTime := time.Now()

		equihashInput, err := d.work.BlockHeader.SerializeAllHeaderBytes()
		if err != nil {
			continue
		}
		equihashInputLog := ""
		for _, e := range equihashInput {
			equihashInputLog = fmt.Sprintf("%s%d", equihashInputLog, int(e))
		}

		minrLog.Infof("EquihashSolveCuda(workId=%d, equihashInput=[%s], nonce=%d, extraNonce=%d)", d.currentWorkID, equihashInputLog, d.work.BlockHeader.Nonce, d.extraNonce)
		C.EquihashSolveCuda(unsafe.Pointer(&equihashInput[0]), C.uint64_t(len(equihashInput)), C.uint32_t(d.work.BlockHeader.Nonce), deviceptr)

		elapsedTime := time.Since(currentTime)
		minrLog.Tracef("GPU #%d: Kernel execution to read time: %v", d.index, elapsedTime)
	}
}

// ListDevices prints a list of CUDA capable GPUs present.
func ListDevices() {
	// CUDA devices
	// Because mumux3/3/cuda/cu likes to panic instead of error.
	defer func() {
		if r := recover(); r != nil {
			fmt.Println("No CUDA Capable GPUs present")
		}
	}()
	devices, _ := getCUDevices()
	for i, dev := range devices {
		fmt.Printf("CUDA Capable GPU #%d: %s\n", i, dev.Name())
	}
}

func NewCuDevice(index int, order int, deviceID cu.Device, workDone chan []byte) (*Device, error) {

	d := &Device{
		index:       index,
		cuDeviceID:  deviceID,
		deviceName:  deviceID.Name(),
		deviceType:  DeviceTypeGPU,
		cuda:        true,
		kind:        DeviceKindNVML,
		quit:        make(chan struct{}),
		newWork:     make(chan *work.Work, 5),
		workDone:    workDone,
		fanPercent:  0,
		temperature: 0,
		tempTarget:  0,
	}

	d.cuInSize = 21

	if !deviceLibraryInitialized {
		err := nvml.Init()
		if err != nil {
			minrLog.Errorf("NVML Init error: %v", err)
		} else {
			deviceLibraryInitialized = true
		}
	}
	fanPercent, temperature := deviceStats(d.index)
	// Newer cards will idle with the fan off so just check if we got
	// a good temperature reading
	if temperature != 0 {
		atomic.StoreUint32(&d.fanPercent, fanPercent)
		atomic.StoreUint32(&d.temperature, temperature)
		d.fanTempActive = true
	}

	// Check if temperature target is specified
	if len(cfg.TempTargetInts) > 0 {
		// Apply the first setting as a global setting
		d.tempTarget = cfg.TempTargetInts[0]

		// Override with the per-device setting if it exists
		for i := range cfg.TempTargetInts {
			if i == order {
				d.tempTarget = uint32(cfg.TempTargetInts[order])
			}
		}
		d.fanControlActive = true
	}

	// validate that we can actually do fan control
	fanControlNotWorking := false
	if d.tempTarget > 0 {
		// validate that fan control is supported
		if !d.fanControlSupported(d.kind) {
			return nil, fmt.Errorf("temperature target of %v for device #%v; "+
				"fan control is not supported on device kind %v", d.tempTarget,
				index, d.kind)
		}
		if !d.fanTempActive {
			minrLog.Errorf("DEV #%d ignoring temperature target of %v; "+
				"could not get initial %v read", index, d.tempTarget, d.kind)
			fanControlNotWorking = true
		}
		if fanControlNotWorking {
			d.tempTarget = 0
			d.fanControlActive = false
		}
	}

	d.started = uint32(time.Now().Unix())

	// Autocalibrate?

	return d, nil
}

func equihashSolutionSize(n, k int) int {
	return 1 << uint32(k) * (n/(k+1) + 1) / 8
}

func deviceStats(index int) (uint32, uint32) {
	fanPercent := uint32(0)
	temperature := uint32(0)

	dh, err := nvml.DeviceGetHandleByIndex(index)
	if err != nil {
		minrLog.Errorf("NVML DeviceGetHandleByIndex error: %v", err)
		return fanPercent, temperature
	}

	nvmlFanSpeed, err := nvml.DeviceFanSpeed(dh)
	if err != nil {
		minrLog.Debugf("NVML DeviceFanSpeed error: %v", err)
	} else {
		fanPercent = uint32(nvmlFanSpeed)
	}

	nvmlTemp, err := nvml.DeviceTemperature(dh)
	if err != nil {
		minrLog.Debugf("NVML DeviceTemperature error: %v", err)
	} else {
		temperature = uint32(nvmlTemp)
	}

	return fanPercent, temperature
}

// unsupported -- just here for compilation
func fanControlSet(index int, fanCur uint32, tempTargetType string, fanChangeLevel string) {
	minrLog.Errorf("NVML fanControl() reached but shouldn't have been")
}

func getInfo() ([]cu.Device, error) {
	cu.Init(0)
	ids := cu.DeviceGetCount()
	minrLog.Infof("%v GPUs", ids)
	var CUdevices []cu.Device
	for i := 0; i < ids; i++ {
		dev := cu.DeviceGet(i)
		CUdevices = append(CUdevices, dev)
		minrLog.Infof("%v: %v", i, dev.Name())
	}
	return CUdevices, nil
}

// getCUDevices returns the list of devices for the given platform.
func getCUDevices() ([]cu.Device, error) {
	cu.Init(0)

	version := cu.Version()
	fmt.Println(version)

	maj := version / 1000
	min := version % 100

	minMajor := 5
	minMinor := 5

	if maj < minMajor || (maj == minMajor && min < minMinor) {
		return nil, fmt.Errorf("Driver does not support CUDA %v.%v API", minMajor, minMinor)
	}

	var numDevices int
	numDevices = cu.DeviceGetCount()
	if numDevices < 1 {
		return nil, fmt.Errorf("No devices found")
	}
	devices := make([]cu.Device, numDevices)
	for i := 0; i < numDevices; i++ {
		dev := cu.DeviceGet(i)
		devices[i] = dev
	}
	return devices, nil
}

func newMinerDevs(m *Miner) (*Miner, int, error) {
	deviceListIndex := 0
	deviceListEnabledCount := 0

	CUdeviceIDs, err := getInfo()
	if err != nil {
		return nil, 0, err
	}

	// XXX Can probably combine these bits with the opencl ones once
	// I decide what to do about the types.

	for _, CUDeviceID := range CUdeviceIDs {
		miningAllowed := false

		// Enforce device restrictions if they exist
		if len(cfg.DeviceIDs) > 0 {
			for _, i := range cfg.DeviceIDs {
				if deviceListIndex == i {
					miningAllowed = true
				}
			}
		} else {
			miningAllowed = true
		}

		if miningAllowed {
			newDevice, err := NewCuDevice(deviceListIndex, deviceListEnabledCount, CUDeviceID, m.workDone)
			deviceListEnabledCount++
			m.devices = append(m.devices, newDevice)
			if err != nil {
				return nil, 0, err
			}
		}
		deviceListIndex++
	}

	return m, deviceListEnabledCount, nil
}

// Return the GPU library in use.
func gpuLib() string {
	return "CUDA"
}
