package stratum

import (
	"strings"
	"sync/atomic"
	"testing"
)

func TestSubmitReplyMayOmitError(t *testing.T) {
	s := &Stratum{submitIDs: []uint64{4}}
	resp, err := s.Unmarshal([]byte(`{"result":true,"id":4}`))
	if err != nil {
		t.Fatal(err)
	}
	s.handleBasicReply(resp)
	if got := atomic.LoadUint64(&s.ValidShares); got != 1 {
		t.Fatalf("valid shares = %d, want 1", got)
	}
	if len(s.submitIDs) != 0 {
		t.Fatalf("submit IDs were not cleared: %v", s.submitIDs)
	}
}

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
