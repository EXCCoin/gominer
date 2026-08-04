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
	"errors"
	"fmt"
	"math/big"
	"runtime"
	"runtime/cgo"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	standalone "github.com/EXCCoin/exccd/blockchain/standalone/v2"
	"github.com/EXCCoin/exccd/wire"

	"github.com/EXCCoin/gominer/util"
	"github.com/EXCCoin/gominer/work"
)

//export equihashProxyGominer
func equihashProxyGominer(userData unsafe.Pointer, solution unsafe.Pointer) C.int {
	w := (*(*cgo.Handle)(userData)).Value().(*eqWorker)
	csol := C.GoBytes(solution, C.int(wire.EquihashSolutionLen))
	w.handleSolution(csol)
	return 0
}

const DeviceTypeGPU = "GPU"

type Device struct {
	// The following variables must only be used atomically.
	fanPercent       uint32
	temperature      uint32
	allDiffOneShares uint64

	// extraNonce is advanced atomically by the workers; the top byte is the
	// device ID (supporting up to 255 devices), the low 3 bytes roll over.
	extraNonce uint32

	sync.Mutex // protects work and hasWork
	work       work.Work
	hasWork    bool

	index int

	deviceName string
	deviceType string

	instances int
	workSize  uint32

	workDone chan WorkResult
	started  uint32
	quit     chan struct{}
	stopOnce sync.Once
}

var errSolverFailed = errors.New("GPU solver failed")

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

func (d *Device) Run() error { return d.runDevice() }

func (d *Device) Stop() {
	d.stopOnce.Do(func() { close(d.quit) })
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
	fanPercent, temperature := backendDeviceStats(d.index)
	atomic.StoreUint32(&d.fanPercent, fanPercent)
	atomic.StoreUint32(&d.temperature, temperature)
}

// Status returns the average solution rate (Sol/s), fan percent, and
// temperature of the device.
func (d *Device) Status() (float64, uint32, uint32) {
	secondsElapsed := uint32(time.Now().Unix()) - d.started
	averageHashRate := float64(0)
	if secondsElapsed != 0 {
		averageHashRate = float64(atomic.LoadUint64(&d.allDiffOneShares)) / float64(secondsElapsed)
	}

	fanPercent := atomic.LoadUint32(&d.fanPercent)
	temperature := atomic.LoadUint32(&d.temperature)

	return averageHashRate, fanPercent, temperature
}

func (d *Device) Release() error { return backendRelease(d.index) }

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

	header, err := hdr.SerializeEquihashHeaderBytes(chainParams.Algorithm(hdr.Height))
	if err != nil {
		minrLog.Errorf("DEV #%d failed to serialize Equihash header: %v", d.index, err)
		return
	}
	if code := C.equihash_verify_c((*C.char)(unsafe.Pointer(&header[0])), C.uint32_t(len(header)),
		(*C.uchar)(unsafe.Pointer(&hdr.EquihashSolution[0]))); code != 0 {
		minrLog.Errorf("DEV #%d rejected invalid Equihash solution (verify code %d)", d.index, int(code))
		return
	}

	solutionType := "block candidate"
	if w.jobID != "" {
		solutionType = "pool share"
	}
	minrLog.Infof("DEV #%d Found %s: hash %s below target %v (height: %d)",
		d.index, solutionType, hashNumBig.String(), hashNum, hdr.Height)
	var buf bytes.Buffer
	if err := hdr.Serialize(&buf); err != nil {
		minrLog.Errorf("Error submitting work: failed to serialize data: %v", err)
		return
	}
	data := make([]byte, work.GetworkDataLen)
	copy(data, buf.Bytes())

	sendOrQuit(d.workDone, WorkResult{data: data, jobID: w.jobID}, d.quit)
}

// runWorker owns one solver instance and grinds nonces on it until shutdown.
func (d *Device) runWorker(id int) error {
	// A dedicated OS thread keeps the GPU context binding stable.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	backendBindThread(d.index)
	solver := C.eq_create(C.uint32_t(d.workSize))
	if solver == nil {
		return fmt.Errorf("DEV #%d: failed to create solver instance %d", d.index, id)
	}
	defer C.eq_destroy(solver)

	w := &eqWorker{d: d, solver: solver, benchmark: cfg.Benchmark}
	handle := cgo.NewHandle(w)
	defer handle.Delete()

	minrLog.Debugf("DEV #%d: solver instance %d running", d.index, id)

	for {
		cw, ok := d.waitForWork()
		if !ok {
			return nil
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
			C.uint32_t(hdr.Nonce), unsafe.Pointer(&handle))
		if n < 0 {
			err := fmt.Errorf("%w: DEV #%d solver instance %d returned %d",
				errSolverFailed, d.index, id, int(n))
			minrLog.Errorf("%v", err)
			return err
		}
		atomic.AddUint64(&d.allDiffOneShares, uint64(n))
	}
}

func runWorkers(count int, stop func(), worker func(int) error) error {
	results := make(chan error, count)
	for id := 0; id < count; id++ {
		go func() { results <- worker(id) }()
	}

	var firstErr error
	for range count {
		if err := <-results; err != nil && firstErr == nil {
			firstErr = err
			stop()
		}
	}
	return firstErr
}

func (d *Device) runDevice() error {
	minrLog.Infof("Started GPU #%d: %s (%d solver instances)", d.index, d.deviceName, d.instances)
	return runWorkers(d.instances, d.Stop, d.runWorker)
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

	d.started = uint32(time.Now().Unix())

	return d, nil
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
			if err != nil {
				return nil, 0, err
			}
			deviceListEnabledCount++
			m.devices = append(m.devices, newDevice)
		}
	}

	return m, deviceListEnabledCount, nil
}

// Return the GPU library in use.
func gpuLib() string {
	return backendName()
}
