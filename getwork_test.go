package main

import (
	"errors"
	"net"
	"testing"
	"time"

	"github.com/EXCCoin/gominer/stratum"
	"github.com/EXCCoin/gominer/work"
)

type shortWriteDeadlineConn struct {
	net.Conn
}

func (c shortWriteDeadlineConn) SetWriteDeadline(deadline time.Time) error {
	if deadline.IsZero() {
		return c.Conn.SetWriteDeadline(deadline)
	}
	return c.Conn.SetWriteDeadline(time.Now().Add(20 * time.Millisecond))
}

func TestGetPoolWorkSubmitWriteTimeoutReleasesLock(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()
	pool := &stratum.Stratum{Conn: shortWriteDeadlineConn{client}}
	done := make(chan error, 1)
	go func() {
		_, err := GetPoolWorkSubmit(make([]byte, work.GetworkDataLen), pool, "job")
		done <- err
	}()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("blocked write succeeded")
		}
	case <-time.After(time.Second):
		t.Fatal("blocked write did not respect its deadline")
	}
	locked := make(chan struct{})
	go func() {
		pool.Lock()
		pool.Unlock()
		close(locked)
	}()
	select {
	case <-locked:
	case <-time.After(time.Second):
		t.Fatal("pool lock remained held after write timeout")
	}
}

func TestMinerReturnsPermanentStratumError(t *testing.T) {
	previousCfg := cfg
	cfg = &config{Pool: "stratum+tcp://pool"}
	defer func() { cfg = previousCfg }()
	want := errors.New("authorization rejected")
	pool := &stratum.Stratum{
		Errors:    make(chan error, 1),
		WorkReady: make(chan struct{}, 1),
	}
	pool.Errors <- want
	m := &Miner{
		pool:             pool,
		quit:             make(chan struct{}),
		workDone:         make(chan WorkResult),
		needsWorkRefresh: make(chan struct{}, 1),
	}
	if err := m.Run(); !errors.Is(err, want) {
		t.Fatalf("error = %v, want %v", err, want)
	}
}
