package stratum

import (
	"strings"
	"testing"
)

func TestNotifySignalsWorkReady(t *testing.T) {
	s := &Stratum{WorkReady: make(chan struct{}, 1)}
	n := NotifyRes{JobID: "1", GenTX1: strings.Repeat("0", 188), Ntime: "00000000"}

	s.handleNotifyRes(n)
	s.handleNotifyRes(n)

	if !s.PoolWork.NewWork {
		t.Fatal("notify did not mark work ready")
	}
	select {
	case <-s.WorkReady:
	default:
		t.Fatal("notify did not wake the miner")
	}
	select {
	case <-s.WorkReady:
		t.Fatal("duplicate notifications should coalesce")
	default:
	}
}
