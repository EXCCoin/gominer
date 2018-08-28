// Copyright (c) 2016 The Decred developers.

package main

import (
	"encoding/hex"
	"sync/atomic"
	"time"

	"github.com/EXCCoin/exccd/blockchain"
	"github.com/EXCCoin/exccd/wire"

	"github.com/EXCCoin/gominer/util"
	"github.com/EXCCoin/gominer/work"
)

var deviceLibraryInitialized = false

// Constants for fan and temperature bits
const (
	ADLFanFailSafe            = uint32(80)
	AMDGPUFanFailSafe         = uint32(204)
	AMDGPUFanMax              = uint32(255)
	AMDTempDivisor            = uint32(1000)
	ChangeLevelNone           = "None"
	ChangeLevelSmall          = "Small"
	ChangeLevelLarge          = "Large"
	DeviceKindAMDGPU          = "AMDGPU"
	DeviceKindADL             = "ADL"
	DeviceKindNVML            = "NVML"
	DeviceKindUnknown         = "Unknown"
	DeviceTypeCPU             = "CPU"
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
	minrLog.Tracef("pre-nonce: %v", hex.EncodeToString(d.work.Data[:]))

	// Bump and set the work ID if the work is new.
	d.currentWorkID++

	// update blockHeader
	blockHeader := wire.BlockHeader{}
	blockHeader.FromBytes(d.work.Data[:])
	d.work.BlockHeader = blockHeader

	d.hasWork = true
}

func (d *Device) Run() {
	err := d.runDevice()
	if err != nil {
		minrLog.Errorf("Error on device: %v", err)
	}
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

	// Hashes that reach this logic and fail the minimal proof of
	// work check are considered to be hardware errors.
	hashNum := d.work.BlockHeader.BlockHash()

	if blockchain.HashToBig(&hashNum).Cmp(blockchain.CompactToBig(d.work.BlockHeader.Bits)) > 0 {
		minrLog.Debugf("DEV #%d Found hash %s above minimum target %s",
			d.index, blockchain.HashToBig(&hashNum).String(), blockchain.CompactToBig(d.work.BlockHeader.Bits).String())
		d.invalidShares++
		return
	}

	d.allDiffOneShares++

	if !cfg.Benchmark {
		// Assess versus the pool or daemon target.
		if blockchain.HashToBig(&hashNum).Cmp(d.work.Target) > 0 {
			minrLog.Debugf("DEV #%d Hash %v bigger than target %032x (boo)",
				d.index, hashNum, d.work.Target.Bytes())
		} else {
			minrLog.Infof("DEV #%d Found hash with work below target! %v (yay)", d.index, hashNum)
			d.validShares++
			//d.workDone <- data // TODO
		}
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

	if fanPercent != 0 || temperature != 0 {
		minrLog.Infof("DEV #%d (%s) %v Fan=%v%% T=%vC",
			d.index,
			d.deviceName,
			util.FormatHashRate(averageHashRate),
			fanPercent,
			temperature)
	} else {
		minrLog.Infof("DEV #%d (%s) %v",
			d.index,
			d.deviceName,
			util.FormatHashRate(averageHashRate),
		)
	}
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
	diffOneShareHashesAvg := uint64(0x00000000FFFFFFFF)

	averageHashRate := (float64(diffOneShareHashesAvg) *
		float64(d.allDiffOneShares)) /
		float64(secondsElapsed)

	fanPercent := atomic.LoadUint32(&d.fanPercent)
	temperature := atomic.LoadUint32(&d.temperature)

	return averageHashRate, fanPercent, temperature
}
