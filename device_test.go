package main

import (
	"encoding/hex"
	"errors"
	"math"
	"math/big"
	"sync/atomic"
	"testing"
	"time"

	"github.com/EXCCoin/exccd/wire"
	"github.com/EXCCoin/gominer/stratum"
	"github.com/EXCCoin/gominer/work"
	"github.com/btcsuite/btclog"
)

func TestStartupStatusRatesAreFinite(t *testing.T) {
	started := uint32(time.Now().Unix())
	d := &Device{started: started}
	rate, _, _ := d.Status()
	if rate != 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
		t.Fatalf("device rate = %v, want 0", rate)
	}

	oldCfg := cfg
	cfg = &config{Pool: "stratum+tcp://pool.example:1234"}
	defer func() { cfg = oldCfg }()
	m := &Miner{started: started, pool: &stratum.Stratum{}}
	_, _, _, _, utility := m.Status()
	if utility != 0 || math.IsNaN(utility) || math.IsInf(utility, 0) {
		t.Fatalf("pool utility = %v, want 0", utility)
	}
}

func TestSendOrQuitStopsBlockedSend(t *testing.T) {
	quit := make(chan struct{})
	done := make(chan bool, 1)
	go func() {
		done <- sendOrQuit(make(chan int), 1, quit)
	}()
	close(quit)

	select {
	case sent := <-done:
		if sent {
			t.Fatal("send succeeded without a receiver")
		}
	case <-time.After(time.Second):
		t.Fatal("send did not stop after quit")
	}
}

func TestHandleSolutionRejectsInvalidProof(t *testing.T) {
	oldLog := minrLog
	minrLog = btclog.Disabled
	defer func() { minrLog = oldLog }()

	results := make(chan WorkResult, 1)
	d := &Device{workDone: results, quit: make(chan struct{})}
	w := eqWorker{
		d:      d,
		header: wire.BlockHeader{Height: 1700000},
		target: new(big.Int).Lsh(big.NewInt(1), 256),
	}

	w.handleSolution(make([]byte, wire.EquihashSolutionLen))

	select {
	case <-results:
		t.Fatal("invalid Equihash proof was submitted")
	default:
	}
}

func TestHandleSolutionAcceptsValidProof(t *testing.T) {
	oldLog := minrLog
	minrLog = btclog.Disabled
	defer func() { minrLog = oldLog }()

	headerBytes, err := hex.DecodeString(
		"050000002a27d50432c7585b1435ca5ec904266a3246040860860fd1c2038e69" +
			"450800006e4497ec71b812ffe8bb51f207b028ad19d8f01a70a2af25b6c8863b" +
			"ece08588afdfc326ac34ec2d2c0e3cadc868d3d5214b7a001334ddcb2ba3001e" +
			"fbf32ca80100385d3d9bee0f05000200f29d0000cf25461e0af5555c00000000" +
			"626909003d11000043b5e6608490310000000000000000003750d39100000000" +
			"0000000000000000000000000000000005000000")
	if err != nil {
		t.Fatal(err)
	}
	solution, err := hex.DecodeString(
		"0389234e87650cbcfc6b50c0e1d876f6b0bd3d53f1159c580e3226af7bf2cc66" +
			"30179b2355d6791ca414cc72636f23afb0d20f6e4798db719e23343afb11f330" +
			"b25fd9fef1d046c9983c222be1193fd0da8bbc23be9b5d74d6f543ed6b65544f" +
			"ccf479d8")
	if err != nil {
		t.Fatal(err)
	}

	var header wire.BlockHeader
	if err := header.FromBytes(append(headerBytes, solution...)); err != nil {
		t.Fatal(err)
	}
	results := make(chan WorkResult, 1)
	w := eqWorker{
		d:      &Device{workDone: results, quit: make(chan struct{})},
		header: header,
		target: new(big.Int).Lsh(big.NewInt(1), 256),
		jobID:  "known-block",
	}
	w.handleSolution(solution)

	select {
	case result := <-results:
		if result.jobID != w.jobID || len(result.data) != work.GetworkDataLen {
			t.Fatalf("unexpected result: job=%q bytes=%d", result.jobID, len(result.data))
		}
	default:
		t.Fatal("valid Equihash proof was rejected")
	}
}

func TestRunWorkersStopsSiblingsOnError(t *testing.T) {
	want := errors.New("solver failed")
	release := make(chan struct{})
	var stops atomic.Int32

	err := runWorkers(3, func() {
		if stops.Add(1) == 1 {
			close(release)
		}
	}, func(id int) error {
		if id == 1 {
			return want
		}
		<-release
		return nil
	})

	if !errors.Is(err, want) {
		t.Fatalf("error = %v, want %v", err, want)
	}
	if got := stops.Load(); got != 1 {
		t.Fatalf("stop calls = %d, want 1", got)
	}
}
