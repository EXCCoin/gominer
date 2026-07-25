package stratum

import (
	"bufio"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"time"
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

func TestReconnectImmediatelySubscribesAndAuthorizes(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	lines := make(chan []string, 1)
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		scanner := bufio.NewScanner(conn)
		got := make([]string, 0, 2)
		for len(got) < 2 && scanner.Scan() {
			got = append(got, scanner.Text())
		}
		lines <- got
	}()

	s := &Stratum{ID: 1}
	s.cfg = Config{Pool: listener.Addr().String(), User: "user", Pass: "pass", Version: "test"}
	start := time.Now()
	if err := s.Reconnect(); err != nil {
		t.Fatal(err)
	}
	defer s.Conn.Close()
	if elapsed := time.Since(start); elapsed >= 2*time.Second {
		t.Fatalf("reconnect handshake took %v", elapsed)
	}

	select {
	case got := <-lines:
		if len(got) != 2 || !strings.Contains(got[0], `"method":"mining.subscribe"`) ||
			!strings.Contains(got[1], `"method":"mining.authorize"`) {
			t.Fatalf("unexpected reconnect handshake: %q", got)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for reconnect handshake")
	}
}
