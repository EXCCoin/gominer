// Copyright (c) 2016 The Decred developers.

package work

import (
	"github.com/EXCCoin/exccd/chaincfg/chainhash"
	"github.com/EXCCoin/exccd/wire"
	"math/big"
)

// These are the locations of various data inside Work.Data.
const (
	TimestampWord  = 2
	Nonce1Word     = 4
	GetworkDataLen = (1 + ((wire.MaxBlockHeaderPayload*8 + 65) / (chainhash.HashBlockSize * 8))) * chainhash.HashBlockSize
)

// NewWork is the constructor for Work.
func NewWork(blockHeader wire.BlockHeader, target *big.Int, jobTime uint32, timeReceived uint32, isGetWork bool, jobID string) *Work {
	return &Work{
		BlockHeader:  blockHeader,
		Target:       target,
		JobTime:      jobTime,
		TimeReceived: timeReceived,
		IsGetWork:    isGetWork,
		JobID:        jobID,
	}
}

// Work holds the data returned from getwork and if needed some stratum related values.
type Work struct {
	BlockHeader  wire.BlockHeader
	Target       *big.Int
	JobTime      uint32
	TimeReceived uint32
	IsGetWork    bool
	JobID        string
}
