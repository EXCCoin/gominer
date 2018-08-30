// Copyright (c) 2016 The Decred developers.

package main

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/EXCCoin/gominer/stratum"
	"github.com/EXCCoin/gominer/work"
)

type Miner struct {
	// The following variables must only be used atomically.
	validShares   uint64
	staleShares   uint64
	invalidShares uint64

	started          uint32
	devices          []*Device
	workDone         chan []byte
	quit             chan struct{}
	needsWorkRefresh chan struct{}
	wg               sync.WaitGroup
	pool             *stratum.Stratum
}

func NewMiner() (*Miner, error) {
	m := &Miner{
		workDone:         make(chan []byte, 10),
		quit:             make(chan struct{}),
		needsWorkRefresh: make(chan struct{}),
	}

	m.devices = make([]*Device, 0)

	// If needed, start pool code.
	if cfg.Pool != "" && !cfg.Benchmark {
		s, err := stratum.StratumConn(cfg.Pool, cfg.PoolUser, cfg.PoolPassword, cfg.Proxy, cfg.ProxyUser, cfg.ProxyPass, version())
		if err != nil {
			return nil, err
		}
		m.pool = s
	}

	m, deviceListEnabledCount, err := newMinerDevs(m)
	if err != nil {
		return nil, err
	}

	if deviceListEnabledCount == 0 {
		return nil, fmt.Errorf("No devices started")
	}

	m.started = uint32(time.Now().Unix())

	return m, nil
}

func (m *Miner) workSubmitThread() {
	defer m.wg.Done()

	for {
		select {
		case <-m.quit:
			return
		case data := <-m.workDone:
			// Only use that is we are not using a pool.
			if m.pool == nil {
				accepted, err := GetWorkSubmit(data)
				if err != nil {
					atomic.AddUint64(&m.invalidShares, 1)
					minrLog.Errorf("Error submitting work: %v", err)
				} else {
					if accepted {
						atomic.AddUint64(&m.validShares, 1)
						minrLog.Debugf("Submitted work successfully: %v", accepted)
					} else {
						atomic.AddUint64(&m.invalidShares, 1)
					}

					m.needsWorkRefresh <- struct{}{}
				}
			} else {
				submitted, err := GetPoolWorkSubmit(data, m.pool)
				if err != nil {
					switch err {
					case stratum.ErrStratumStaleWork:
						atomic.AddUint64(&m.staleShares, 1)
						minrLog.Debugf("Share submitted to pool was stale")

					default:
						atomic.AddUint64(&m.invalidShares, 1)
						minrLog.Errorf("Error submitting work to pool: %v", err)
					}
				} else {
					if submitted {
						minrLog.Debugf("Submitted work to pool successfully: %v", submitted)
					}
					m.needsWorkRefresh <- struct{}{}
				}
			}
		}
	}
}

func (m *Miner) workRefreshThread() {
	defer m.wg.Done()

	t := time.NewTicker(5 * time.Second)
	defer t.Stop()

	for {
		// Only use that is we are not using a pool.
		if m.pool == nil {
			w, err := GetWork()
			if err != nil {
				minrLog.Errorf("Error in getwork: %v", err)
			} else {
				for _, d := range m.devices {
					d.SetWork(w)
				}
			}
		} else {
			m.pool.Lock()
			if m.pool.PoolWork.NewWork {
				w, err := GetPoolWork(m.pool)
				m.pool.Unlock()
				if err != nil {
					minrLog.Errorf("Error in getpoolwork: %v", err)
				} else {
					for _, d := range m.devices {
						d.SetWork(w)
					}
				}
			} else {
				m.pool.Unlock()
			}
		}
		select {
		case <-m.quit:
			return
		case <-t.C:
		case <-m.needsWorkRefresh:
		}
	}
}

func (m *Miner) printStatsThread() {
	defer m.wg.Done()

	t := time.NewTicker(time.Second * 30)
	defer t.Stop()

	for {
		select {
		case <-m.quit:
			return
		case <-t.C:
		}

		if !cfg.Benchmark {
			valid, rejected, stale, total, utility := m.Status()

			if cfg.Pool != "" {
				minrLog.Infof("Global stats: Accepted: %v, Rejected: %v, Stale: %v, Total: %v",
					valid,
					rejected,
					stale,
					total,
				)
				secondsElapsed := uint32(time.Now().Unix()) - m.started
				if (secondsElapsed / 60) > 0 {
					minrLog.Infof("Global utility (accepted shares/min): %v", utility)
				}
			} else {
				minrLog.Infof("Global stats: Accepted: %v, Rejected: %v, Total: %v",
					valid,
					rejected,
					total,
				)
			}
		}

		for _, d := range m.devices {
			d.UpdateFanTemp()
			d.PrintStats()
			if d.fanControlActive {
				d.fanControl()
			}
		}
	}
}

func (m *Miner) Run() {
	m.wg.Add(len(m.devices))

	for _, d := range m.devices {
		device := d
		go func() {
			device.Run()
			device.Release()
			m.wg.Done()
		}()
	}

	m.wg.Add(1)
	go m.workSubmitThread()

	if cfg.Benchmark {
		minrLog.Warn("Running in BENCHMARK mode! No real mining taking place!")
		w := &work.Work{}
		w.BlockHeader.FromBytes([]byte{6, 0, 0, 0, 254, 122, 189, 44, 76, 62, 223, 111, 1, 219, 247, 151, 98, 58, 211,
		148, 251, 40, 66, 90, 218, 43, 222, 56, 253, 180, 154, 189, 95, 114, 246, 4, 231, 184, 29, 242, 25, 197, 18,
		189, 140, 31, 202, 156, 190, 237, 126, 234, 83, 181, 27, 202, 76, 218, 38, 62, 123, 10, 80, 159, 223, 150, 171,
		248, 32, 145, 190, 102, 57, 158, 92, 49, 17, 191, 198, 84, 55, 118, 147, 144, 125, 169, 98, 216, 144, 44, 14,
		159, 205, 124, 81, 225, 50, 42, 211, 148, 1, 0, 2, 192, 167, 171, 59, 58, 5, 0, 0, 0, 232, 17, 0, 0, 22, 238,
		23, 32, 122, 215, 57, 3, 2, 0, 0, 0, 20, 77, 0, 0, 74, 19, 0, 0, 16, 200, 135, 91, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 8, 192})
		for _, d := range m.devices {
			d.SetWork(w)
		}
	} else {
		m.wg.Add(1)
		go m.workRefreshThread()
	}

	m.wg.Add(1)
	go m.printStatsThread()

	m.wg.Wait()
}

func (m *Miner) Stop() {
	close(m.quit)
	for _, d := range m.devices {
		d.Stop()
	}
}

func (m *Miner) Status() (uint64, uint64, uint64, uint64, float64) {
	if cfg.Pool != "" {
		valid := atomic.LoadUint64(&m.pool.ValidShares)
		rejected := atomic.LoadUint64(&m.pool.InvalidShares)
		stale := atomic.LoadUint64(&m.staleShares)
		total := valid + rejected + stale

		secondsElapsed := uint32(time.Now().Unix()) - m.started
		utility := float64(valid) / (float64(secondsElapsed) / float64(60))

		return valid, rejected, stale, total, utility
	}

	valid := atomic.LoadUint64(&m.validShares)
	rejected := atomic.LoadUint64(&m.invalidShares)
	total := valid + rejected

	return valid, rejected, 0, total, 0
}
