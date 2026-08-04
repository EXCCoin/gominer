package util

import (
	"math"
	"math/big"
	"testing"
)

func TestReverseToIntParsesHexVersion(t *testing.T) {
	got, err := ReverseToInt("efbeadde")
	if err != nil {
		t.Fatal(err)
	}
	if uint32(got) != 0xdeadbeef {
		t.Fatalf("version = %08x, want deadbeef", uint32(got))
	}
}

func TestDiffToTargetRejectsNonFiniteAndOverflow(t *testing.T) {
	for _, diff := range []float64{math.NaN(), math.Inf(1), math.MaxFloat64} {
		if _, err := DiffToTarget(diff, big.NewInt(1)); err == nil {
			t.Fatalf("difficulty %v was accepted", diff)
		}
	}
}
