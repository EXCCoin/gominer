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

func TestReconnectWaitsForFreshWork(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()

	lines := make(chan []string, 1)
	sendNotify := make(chan struct{})
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
		for _, response := range []string{
			`{"id":1,"result":[[["mining.set_difficulty","1"],["mining.notify","session"]],"0000000000000000",12],"error":null}`,
			`{"id":2,"result":true,"error":null}`,
			`{"id":null,"method":"mining.set_difficulty","params":[1024]}`,
		} {
			_, _ = conn.Write([]byte(response + "\n"))
		}
		lines <- got
		<-sendNotify
		notify := `{"id":null,"method":"mining.notify","params":["job","` +
			strings.Repeat("0", 64) + `","` + strings.Repeat("0", 188) +
			`","",[],"01000000","1a12334a","00000000",true]}`
		_, _ = conn.Write([]byte(notify + "\n"))
	}()

	s := &Stratum{ID: 1, WorkReady: make(chan struct{}, 1)}
	s.cfg = Config{Pool: listener.Addr().String(), User: "user", Pass: "pass", Version: "test"}
	done := make(chan error, 1)
	go func() { done <- s.Reconnect() }()

	select {
	case got := <-lines:
		if len(got) != 2 || !strings.Contains(got[0], `"method":"mining.subscribe"`) ||
			!strings.Contains(got[1], `"method":"mining.authorize"`) {
			t.Fatalf("unexpected reconnect handshake: %q", got)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for reconnect handshake")
	}
	select {
	case err := <-done:
		t.Fatalf("reconnect returned before fresh work: %v", err)
	case <-time.After(100 * time.Millisecond):
	}
	close(sendNotify)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("reconnect did not accept fresh work")
	}
	defer s.Conn.Close()
	if !s.PoolWork.NewWork {
		t.Fatal("fresh work was not marked ready")
	}
	select {
	case <-s.WorkReady:
	default:
		t.Fatal("fresh work did not wake the miner")
	}
}
