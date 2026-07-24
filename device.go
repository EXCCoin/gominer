// Copyright (c) 2016 The Decred developers.

package main

/*
#cgo CFLAGS: -O3 -Wall -Werror
#include "eqcuda1445/eqcuda1445.h"

static int eqSolveGo(EqSolver *s, const void *hdr, uint32_t len, uint32_t nonce, void *ud) __attribute__((unused));
static int eqSolveGo(EqSolver *s, const void *hdr, uint32_t len, uint32_t nonce, void *ud) {
	return eq_solve(s, hdr, len, nonce, equihashProxyGominer, ud);
}
*/
import "C"
import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math/big"
	"runtime"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	standalone "github.com/EXCCoin/exccd/blockchain/standalone/v2"
	"github.com/EXCCoin/exccd/wire"

	"github.com/EXCCoin/gominer/util"
	"github.com/EXCCoin/gominer/work"

	cptr "github.com/mattn/go-pointer"
)

//export equihashProxyGominer
func equihashProxyGominer(userData unsafe.Pointer, solution unsafe.Pointer) C.int {
	w := cptr.Restore(userData).(*eqWorker)
	csol := C.GoBytes(solution, C.int(wire.EquihashSolutionLen))
	w.handleSolution(csol)
	return 0
}

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
	fanPercent       uint32
	temperature      uint32
	allDiffOneShares uint64
	validShares      uint64
	invalidShares    uint64

	// extraNonce is advanced atomically by the workers; the top byte is the
	// device ID (supporting up to 255 devices), the low 3 bytes roll over.
	extraNonce uint32

	sync.Mutex // protects work and hasWork
	work       work.Work
	hasWork    bool

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

	instances int
	workSize  uint32

	workDone chan WorkResult
	started  uint32
	quit     chan struct{}
}

// eqWorker is one solver instance on a device. Each worker owns its own GPU
// buffers and remembers the exact header it is solving so that concurrent
// workers never submit a header they did not solve.
type eqWorker struct {
	d         *Device
	solver    *C.EqSolver
	header    wire.BlockHeader
	target    *big.Int
	jobID     string
	benchmark bool
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
	d.Lock()
	d.work = *w
	d.hasWork = true
	d.Unlock()
}

// waitForWork returns a snapshot of the current work, blocking until work is
// available. ok is false when the device is shutting down.
func (d *Device) waitForWork() (w work.Work, ok bool) {
	for {
		select {
		case <-d.quit:
			return work.Work{}, false
		default:
		}
		d.Lock()
		if d.hasWork {
			w = d.work
			d.Unlock()
			return w, true
		}
		d.Unlock()
		time.Sleep(100 * time.Millisecond)
	}
}

func (d *Device) nextExtraNonce() uint32 {
	for {
		old := atomic.LoadUint32(&d.extraNonce)
		next := old
		util.RolloverExtraNonce(&next)
		if atomic.CompareAndSwapUint32(&d.extraNonce, old, next) {
			return next
		}
	}
}

func (d *Device) PrintStats() {
	secondsElapsed := uint32(time.Now().Unix()) - d.started
	if secondsElapsed == 0 {
		return
	}

	averageHashRate, fanPercent, temperature := d.Status()
	log := fmt.Sprintf("DEV #%d (%s) %v (solutions=%d)", d.index, d.deviceName,
		util.FormatHashRate(averageHashRate), atomic.LoadUint64(&d.allDiffOneShares))

	if fanPercent != 0 {
		log = fmt.Sprintf("%s (Fan=%v%%)", log, fanPercent)
	}

	if temperature != 0 {
		log = fmt.Sprintf("%s (T=%vC)", log, temperature)
	}

	minrLog.Info(log)
}

// UpdateFanTemp updates a device's statistics
func (d *Device) UpdateFanTemp() {
	if d.fanTempActive {
		switch d.kind {
		case DeviceKindADL, DeviceKindAMDGPU, DeviceKindNVML:
			fanPercent, temperature := backendDeviceStats(d.index)
			atomic.StoreUint32(&d.fanPercent, fanPercent)
			atomic.StoreUint32(&d.temperature, temperature)
		}
	}
}

// Status returns the average solution rate (Sol/s), fan percent, and
// temperature of the device.
func (d *Device) Status() (float64, uint32, uint32) {
	secondsElapsed := uint32(time.Now().Unix()) - d.started

	averageHashRate := float64(atomic.LoadUint64(&d.allDiffOneShares)) / float64(secondsElapsed)

	fanPercent := atomic.LoadUint32(&d.fanPercent)
	temperature := atomic.LoadUint32(&d.temperature)

	return averageHashRate, fanPercent, temperature
}

func (d *Device) Release() {
	backendRelease(d.index)
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

// handleSolution is invoked (via the cgo proxy) for every solution the GPU
// found for the worker's current header.
func (w *eqWorker) handleSolution(solution []byte) {
	if w.benchmark {
		return
	}

	d := w.d
	hdr := w.header // copy; the worker may already be reused for new work
	copy(hdr.EquihashSolution[:], solution)

	hashNum := hdr.BlockHash()
	hashNumBig := standalone.HashToBig(&hashNum)

	// Assess versus the pool or daemon target.
	if hashNumBig.Cmp(w.target) > 0 {
		minrLog.Debugf("DEV #%d Hash %s bigger than target %032x (boo)", d.index, hashNumBig, w.target.Bytes())
		return
	}

	solutionType := "block candidate"
	if w.jobID != "" {
		solutionType = "pool share"
	}
	minrLog.Infof("DEV #%d Found %s: hash %s below target %v (height: %d)",
		d.index, solutionType, hashNumBig.String(), hashNum, hdr.Height)
	atomic.AddUint64(&d.validShares, 1)

	var buf bytes.Buffer
	if err := hdr.Serialize(&buf); err != nil {
		minrLog.Errorf("Error submitting work: failed to serialize data: %v", err)
		return
	}
	data := make([]byte, work.GetworkDataLen)
	copy(data, buf.Bytes())

	d.workDone <- WorkResult{data: data, jobID: w.jobID}
}

// runWorker owns one solver instance and grinds nonces on it until shutdown.
func (d *Device) runWorker(id int) {
	// A dedicated OS thread keeps the GPU context binding stable.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	backendBindThread(d.index)
	solver := C.eq_create(C.uint32_t(d.workSize))
	if solver == nil {
		minrLog.Errorf("DEV #%d: failed to create solver instance %d", d.index, id)
		return
	}
	defer C.eq_destroy(solver)

	w := &eqWorker{d: d, solver: solver, benchmark: cfg.Benchmark}
	ptr := cptr.Save(w)
	defer cptr.Unref(ptr)

	minrLog.Debugf("DEV #%d: solver instance %d running", d.index, id)

	for {
		cw, ok := d.waitForWork()
		if !ok {
			return
		}

		hdr := cw.BlockHeader

		// Distinct extraNonce per attempt guarantees workers never grind
		// identical inputs.
		extraNonce := d.nextExtraNonce()
		binary.LittleEndian.PutUint64(hdr.ExtraData[:], uint64(extraNonce))

		// Only solo work allows rolling the timestamp.
		ts := cw.JobTime
		if cw.IsGetWork {
			ts = cw.JobTime + (uint32(time.Now().Unix()) - cw.TimeReceived)
		}
		hdr.Timestamp = time.Unix(int64(ts), 0)

		nonce, err := wire.RandomUint64()
		if err != nil {
			minrLog.Errorf("Unexpected error while generating random nonce: %v", err)
			nonce = uint64(extraNonce)
		}
		hdr.Nonce = uint32(nonce)

		w.header = hdr
		w.target = cw.Target
		w.jobID = cw.JobID

		algo := chainParams.Algorithm(hdr.Height)
		input, err := hdr.SerializeEquihashHeaderBytes(algo)
		if err != nil {
			minrLog.Errorf("DEV #%d: failed to serialize equihash header: %v", d.index, err)
			continue
		}

		n := C.eqSolveGo(solver, unsafe.Pointer(&input[0]), C.uint32_t(len(input)),
			C.uint32_t(hdr.Nonce), ptr)
		if n < 0 {
			minrLog.Errorf("DEV #%d: solver instance %d CUDA error %d", d.index, id, int(n))
			return
		}
		atomic.AddUint64(&d.allDiffOneShares, uint64(n))
	}
}

func (d *Device) runDevice() error {
	minrLog.Infof("Started GPU #%d: %s (%d solver instances)", d.index, d.deviceName, d.instances)

	var wg sync.WaitGroup
	for i := 0; i < d.instances; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			d.runWorker(id)
		}(i)
	}
	wg.Wait()
	return nil
}

// ListDevices prints a list of capable GPUs present.
func ListDevices() {
	// Because the cu wrappers panic instead of returning errors.
	defer func() {
		if r := recover(); r != nil {
			fmt.Printf("No %s capable GPUs present\n", backendName())
		}
	}()
	names, err := backendEnumerate()
	if err != nil {
		fmt.Printf("No %s capable GPUs present: %v\n", backendName(), err)
		return
	}
	for i, name := range names {
		fmt.Printf("%s capable GPU #%d: %s\n", backendName(), i, name)
	}
}

func NewDevice(index int, order int, name string, workDone chan WorkResult) (*Device, error) {
	d := &Device{
		index:      index,
		deviceName: name,
		deviceType: DeviceTypeGPU,
		cuda:       true,
		kind:       DeviceKindNVML,
		quit:       make(chan struct{}),
		workDone:   workDone,
		extraNonce: uint32(index) << 24,
	}

	// Per-device work size (solver thread count); 0 lets the solver scale
	// to the GPU.
	if len(cfg.WorkSizeInts) > 0 {
		d.workSize = cfg.WorkSizeInts[0]
		if order < len(cfg.WorkSizeInts) {
			d.workSize = cfg.WorkSizeInts[order]
		}
	}

	// Number of concurrent solver instances.
	d.instances = cfg.Instances
	if d.instances <= 0 {
		d.instances = backendAutoInstances(index)
	}

	fanPercent, temperature := backendDeviceStats(d.index)
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
		if order < len(cfg.TempTargetInts) {
			d.tempTarget = cfg.TempTargetInts[order]
		}
		d.fanControlActive = true
	}

	// validate that we can actually do fan control
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
			d.tempTarget = 0
			d.fanControlActive = false
		}
	}

	d.started = uint32(time.Now().Unix())

	return d, nil
}

// unsupported -- just here for compilation
func fanControlSet(index int, fanCur uint32, tempTargetType string, fanChangeLevel string) {
	minrLog.Errorf("fanControl() reached but shouldn't have been")
}

func newMinerDevs(m *Miner) (*Miner, int, error) {
	deviceListEnabledCount := 0

	names, err := backendEnumerate()
	if err != nil {
		return nil, 0, err
	}
	minrLog.Infof("%v GPUs", len(names))

	for deviceListIndex, name := range names {
		minrLog.Infof("%v: %v", deviceListIndex, name)
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
			newDevice, err := NewDevice(deviceListIndex, deviceListEnabledCount, name, m.workDone)
			deviceListEnabledCount++
			m.devices = append(m.devices, newDevice)
			if err != nil {
				return nil, 0, err
			}
		}
	}

	return m, deviceListEnabledCount, nil
}

// Return the GPU library in use.
func gpuLib() string {
	return backendName()
}
