// Copyright (c) 2016 The Decred developers.

package work

import (
	"math/big"
	"github.com/EXCCoin/exccd/wire"
)

// These are the locations of various data inside Work.Data.
const (
	TimestampWord = 2
	Nonce0Word    = 3
	Nonce1Word    = 4
	Nonce2Word    = 5
)

// NewWork is the constructor for Work.
func NewWork(data [320]byte, blockHeader wire.BlockHeader, target *big.Int, jobTime uint32, timeReceived uint32,
	isGetWork bool) *Work {
	return &Work{
		Data:         data,
		BlockHeader:  blockHeader,
		Target:       target,
		JobTime:      jobTime,
		TimeReceived: timeReceived,
		IsGetWork:    isGetWork,
	}
}

// Work holds the data returned from getwork and if needed some stratum related
// values.
type Work struct {
	Data         [320]byte
	BlockHeader  wire.BlockHeader
	Target       *big.Int
	JobTime      uint32
	TimeReceived uint32
	IsGetWork    bool
}
