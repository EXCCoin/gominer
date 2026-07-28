package main

import (
	"errors"
	"sync/atomic"
	"testing"
)

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
