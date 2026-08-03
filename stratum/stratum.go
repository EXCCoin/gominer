// Copyright (c) 2016 The Decred developers.

package stratum

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/btcsuite/go-socks/socks"
	"github.com/davecgh/go-spew/spew"

	"github.com/EXCCoin/exccd/chaincfg/v3"
	"github.com/EXCCoin/exccd/wire"

	"github.com/EXCCoin/gominer/util"
	"github.com/EXCCoin/gominer/work"
)

var chainParams = chaincfg.MainNetParams()

const reconnectRetryDelay = 5 * time.Second

const (
	maxStratumMessageSize = 1 << 20
	minCoinbase1Size      = 108
)

// ErrStratumStaleWork indicates that the work to send to the pool was stale.
var ErrStratumStaleWork = fmt.Errorf("Stale work, throwing away")

var (
	errAuthorizationRejected = errors.New("stratum authorization rejected")
	errReconnectRequested    = errors.New("stratum server requested reconnect")
	errStratumMessageTooLong = errors.New("stratum message exceeds 1 MiB")
)

// Stratum holds all the shared information for a stratum connection.
// XXX most of these should be unexported and use getters/setters.
type Stratum struct {
	// The following variables must only be used atomically.
	ValidShares   uint64
	InvalidShares uint64
	latestJobTime uint32

	sync.Mutex
	cfg       Config
	Conn      net.Conn
	Reader    *bufio.Reader
	ID        uint64
	authID    uint64
	subID     uint64
	submitIDs []uint64
	Diff      float64
	Target    *big.Int
	PoolWork  NotifyWork
	WorkReady chan struct{}
	Errors    chan error

	Started uint32
}

// Config holdes the config options that may be used by a stratum pool.
type Config struct {
	Pool      string
	User      string
	Pass      string
	Proxy     string
	ProxyUser string
	ProxyPass string
	Version   string
}

// NotifyWork holds all the info recieved from a mining.notify message along
// with the Work data generate from it.
type NotifyWork struct {
	Clean             bool
	ExtraNonce1       string
	ExtraNonce2       uint64
	ExtraNonce2Length float64
	Nonce2            uint32
	CB1               string
	CB2               string
	Height            int64
	NtimeDelta        int64
	JobID             string
	Hash              string
	Nbits             string
	Ntime             string
	Version           string
	NewWork           bool
	Work              *work.Work
}

// StratumMsg is the basic message object from stratum.
type StratumMsg struct {
	Method string `json:"method"`
	// Need to make generic.
	Params []string    `json:"params"`
	ID     interface{} `json:"id"`
}

// StratumRsp is the basic response type from stratum.
type StratumRsp struct {
	Method string `json:"method"`
	// Need to make generic.
	ID     interface{}      `json:"id"`
	Error  StratErr         `json:"error,omitempty"`
	Result *json.RawMessage `json:"result,omitempty"`
}

// StratErr is the basic error type (a number and a string) sent by
// the stratum server.
type StratErr struct {
	ErrNum uint64
	ErrStr string
	Result *json.RawMessage `json:"result,omitempty"`
}

type PoolError struct {
	Code    uint64 `json:"code"`
	Message string `json:"message"`
	Data    string `json:"data,omitempty"`
}

// Basic reply is a reply type for any of the simple messages.
type BasicReply struct {
	ID     interface{} `json:"id"`
	Error  StratErr    `json:"error,omitempty"`
	Result bool        `json:"result"`
}

// SubscribeReply models the server response to a subscribe message.
type SubscribeReply struct {
	SubscribeID       string
	ExtraNonce1       string
	ExtraNonce2Length float64
}

// NotifyRes models the json from a mining.notify message.
type NotifyRes struct {
	JobID          string
	Hash           string
	GenTX1         string
	GenTX2         string
	MerkleBranches []string
	BlockVersion   string
	Nbits          string
	Ntime          string
	CleanJobs      bool
}

func validateNotify(n NotifyRes) error {
	if n.JobID == "" {
		return errors.New("notify has no job ID")
	}
	cb1, err := hex.DecodeString(n.GenTX1)
	if err != nil {
		return fmt.Errorf("invalid coinbase part 1: %w", err)
	}
	if len(cb1) < minCoinbase1Size {
		return fmt.Errorf("coinbase part 1 is %d bytes, need at least %d", len(cb1), minCoinbase1Size)
	}
	fields := []struct {
		name  string
		value string
		size  int
	}{
		{"previous hash", n.Hash, 32},
		{"coinbase part 2", n.GenTX2, -1},
		{"block version", n.BlockVersion, 4},
		{"difficulty bits", n.Nbits, 4},
		{"timestamp", n.Ntime, 4},
	}
	for _, field := range fields {
		decoded, err := hex.DecodeString(field.value)
		if err != nil {
			return fmt.Errorf("invalid %s: %w", field.name, err)
		}
		if field.size >= 0 && len(decoded) != field.size {
			return fmt.Errorf("%s is %d bytes, need %d", field.name, len(decoded), field.size)
		}
	}
	return nil
}

// Submit models a submission message.
type Submit struct {
	Params []string    `json:"params"`
	ID     interface{} `json:"id"`
	Method string      `json:"method"`
}

// errJsonType is an error for json that we do not expect.
var errJsonType = errors.New("Unexpected type in json.")

func sliceContains(s []uint64, e uint64) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

func sliceRemove(s []uint64, e uint64) []uint64 {
	for i, a := range s {
		if a == e {
			return append(s[:i], s[i+1:]...)
		}
	}

	return s
}

func unmarshalPoolError(raw json.RawMessage) (StratErr, error) {
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return StratErr{}, nil
	}
	if raw[0] == '[' {
		var fields []json.RawMessage
		if err := json.Unmarshal(raw, &fields); err != nil {
			return StratErr{}, err
		}
		if len(fields) < 2 {
			return StratErr{}, errJsonType
		}
		var result StratErr
		if err := json.Unmarshal(fields[0], &result.ErrNum); err != nil {
			return StratErr{}, err
		}
		if err := json.Unmarshal(fields[1], &result.ErrStr); err != nil {
			return StratErr{}, err
		}
		return result, nil
	}

	var poolErr struct {
		Code    uint64 `json:"code"`
		Message string `json:"message"`
	}
	if err := json.Unmarshal(raw, &poolErr); err != nil {
		return StratErr{}, err
	}
	return StratErr{ErrNum: poolErr.Code, ErrStr: poolErr.Message}, nil
}

func unmarshalBasicReply(objmap map[string]json.RawMessage) (*BasicReply, error) {
	var id uint64
	if err := json.Unmarshal(objmap["id"], &id); err != nil {
		return nil, err
	}
	var result bool
	rawResult, hasResult := objmap["result"]
	if !hasResult || bytes.Equal(rawResult, []byte("null")) {
		if rawError, ok := objmap["error"]; !ok || bytes.Equal(rawError, []byte("null")) {
			return nil, errJsonType
		}
	} else if err := json.Unmarshal(rawResult, &result); err != nil {
		return nil, err
	}

	resp := &BasicReply{ID: id, Result: result}
	if result {
		return resp, nil
	}
	if raw, ok := objmap["error"]; ok {
		poolErr, err := unmarshalPoolError(raw)
		if err != nil {
			return nil, err
		}
		resp.Error = poolErr
	}
	return resp, nil
}

// StratumConn starts the initial connection to a stratum pool and sets defaults
// in the pool object.
func StratumConn(pool, user, pass, proxy, proxyUser, proxyPass, version string) (*Stratum, error) {
	stratum := Stratum{
		WorkReady: make(chan struct{}, 1),
		Errors:    make(chan error, 1),
	}
	stratum.cfg.User = user
	stratum.cfg.Pass = pass
	stratum.cfg.Proxy = proxy
	stratum.cfg.ProxyUser = proxyUser
	stratum.cfg.ProxyPass = proxyPass
	stratum.cfg.Version = version

	log.Infof("Using pool: %v", pool)
	proto := "stratum+tcp://"
	if strings.HasPrefix(pool, proto) {
		pool = strings.Replace(pool, proto, "", 1)
	} else {
		err := errors.New("Only stratum pools supported.")
		return nil, err
	}
	stratum.ID = 1
	stratum.cfg.Pool = pool

	// We will set it for sure later but this really should be the value and
	// setting it here will prevent so incorrect matches based on the
	// default 0 value.
	stratum.authID = 2

	// Target for share is 1 unless we hear otherwise.
	stratum.Diff = 1
	var err error
	stratum.Target, err = util.DiffToTarget(stratum.Diff, chainParams.PowLimit)
	if err != nil {
		return nil, err
	}
	stratum.PoolWork.NewWork = false
	if err := stratum.Reconnect(); err != nil {
		return nil, err
	}
	go stratum.Listen()

	return &stratum, nil
}

// Reconnect reconnects to a stratum server if the connection has been lost.
func (s *Stratum) Reconnect() error {
	for {
		err := s.reconnect()
		if errors.Is(err, errReconnectRequested) {
			continue
		}
		return err
	}
}

func (s *Stratum) reconnect() error {
	s.Lock()
	if s.Conn != nil {
		_ = s.Conn.Close()
	}
	pool := s.cfg.Pool
	proxyAddr := s.cfg.Proxy
	proxyUser := s.cfg.ProxyUser
	proxyPass := s.cfg.ProxyPass
	s.Unlock()

	var conn net.Conn
	var err error
	if proxyAddr != "" {
		proxy := &socks.Proxy{
			Addr:     proxyAddr,
			Username: proxyUser,
			Password: proxyPass,
		}
		conn, err = proxy.DialTimeout("tcp", pool, reconnectRetryDelay)
	} else {
		conn, err = net.DialTimeout("tcp", pool, reconnectRetryDelay)
	}
	if err != nil {
		return err
	}

	s.Lock()
	s.Conn = conn
	s.Reader = bufio.NewReaderSize(conn, maxStratumMessageSize)
	s.PoolWork.NewWork = false
	s.submitIDs = nil
	atomic.StoreUint32(&s.latestJobTime, 0)
	err = s.sendHandshake()
	s.Unlock()
	if err != nil {
		_ = conn.Close()
		return err
	}
	if err := s.waitHandshake(conn); err != nil {
		_ = conn.Close()
		return err
	}

	s.Lock()
	s.Started = uint32(time.Now().Unix())
	s.Unlock()
	return nil
}

func (s *Stratum) handshake(conn net.Conn) error {
	s.Lock()
	err := s.sendHandshake()
	s.Unlock()
	if err != nil {
		return err
	}
	return s.waitHandshake(conn)
}

func (s *Stratum) sendHandshake() error {
	if err := s.subscribe(); err != nil {
		return fmt.Errorf("subscribe: %w", err)
	}
	if err := s.auth(); err != nil {
		return fmt.Errorf("authorize: %w", err)
	}
	return nil
}

func (s *Stratum) waitHandshake(conn net.Conn) error {
	var subscribed, authorized bool
	var notify *NotifyRes
	handshakeDeadline := time.Now().Add(reconnectRetryDelay)
	var firstWorkDeadline time.Time
	for !subscribed || !authorized || notify == nil {
		deadline := handshakeDeadline
		if subscribed && authorized {
			if firstWorkDeadline.IsZero() {
				firstWorkDeadline = time.Now().Add(4 * chainParams.TargetTimePerBlock)
			}
			deadline = firstWorkDeadline
		}
		if err := conn.SetReadDeadline(deadline); err != nil {
			return err
		}
		result, err := readStratumMessage(s.Reader)
		if err != nil {
			return fmt.Errorf("waiting for fresh work: %w", err)
		}

		log.Debug(strings.TrimSuffix(string(result), "\n"))
		resp, err := s.Unmarshal(result)
		if err != nil {
			log.Error(err)
			continue
		}
		switch r := resp.(type) {
		case *BasicReply:
			s.handleResponse(resp)
			id := r.ID.(uint64)
			switch id {
			case s.authID:
				if !r.Result {
					return fmt.Errorf("%w: %s", errAuthorizationRejected, r.Error.ErrStr)
				}
				authorized = true
			case s.subID:
				if !r.Result {
					return fmt.Errorf("subscription rejected: %s", r.Error.ErrStr)
				}
			}
		case *SubscribeReply:
			subscribed = true
			s.handleResponse(resp)
		case NotifyRes:
			n := r
			notify = &n
		case StratumMsg:
			if r.Method == "client.reconnect" {
				if err := s.applyReconnect(r); err != nil {
					return err
				}
				return errReconnectRequested
			}
			s.handleResponse(resp)
		default:
			s.handleResponse(resp)
		}
	}
	s.handleResponse(*notify)
	if err := conn.SetReadDeadline(time.Time{}); err != nil {
		return err
	}
	return nil
}

func (s *Stratum) reconnectUntilReady() error {
	for {
		if err := s.Reconnect(); err != nil {
			if errors.Is(err, errAuthorizationRejected) {
				return err
			}
			log.Errorf("Reconnect failed: %v. Retrying in %v.", err, reconnectRetryDelay)
			time.Sleep(reconnectRetryDelay)
			continue
		}
		log.Info("Reconnected.")
		return nil
	}
}

func readStratumMessage(reader *bufio.Reader) ([]byte, error) {
	message, err := reader.ReadSlice('\n')
	if errors.Is(err, bufio.ErrBufferFull) {
		return nil, errStratumMessageTooLong
	}
	return message, err
}

// Listen is the listener for the incoming messages from the stratum pool.
func (s *Stratum) Listen() {
	log.Debug("Starting Listener")

	for {
		result, err := readStratumMessage(s.Reader)
		if err != nil {
			log.Errorf("Connection lost: %v. Reconnecting.", err)
			if err := s.reconnectUntilReady(); err != nil {
				select {
				case s.Errors <- err:
				default:
				}
				return
			}
			continue
		}

		log.Debug(strings.TrimSuffix(string(result), "\n"))
		resp, err := s.Unmarshal(result)
		if err != nil {
			log.Error(err)
			continue
		}

		if err := s.handleResponse(resp); err != nil {
			select {
			case s.Errors <- err:
			default:
			}
			return
		}
	}
}

func (s *Stratum) handleResponse(resp interface{}) error {
	switch resp.(type) {
	case *BasicReply:
		s.handleBasicReply(resp)
	case StratumMsg:
		return s.handleStratumMsg(resp)
	case NotifyRes:
		s.handleNotifyRes(resp)
	case *SubscribeReply:
		s.handleSubscribeReply(resp)
	default:
		log.Info("Unhandled message: ", resp)
	}
	return nil
}

func (s *Stratum) handleBasicReply(resp interface{}) {
	s.Lock()
	defer s.Unlock()
	aResp := resp.(*BasicReply)

	if int(aResp.ID.(uint64)) == int(s.authID) {
		if aResp.Result {
			log.Debug("Logged in")
		} else {
			log.Error("Auth failure.")
		}
	}
	if sliceContains(s.submitIDs, aResp.ID.(uint64)) {
		if aResp.Result {
			atomic.AddUint64(&s.ValidShares, 1)
			log.Debug("Share accepted")
		} else {
			atomic.AddUint64(&s.InvalidShares, 1)
			log.Error("Share rejected: ", aResp.Error.ErrStr)
		}
		s.submitIDs = sliceRemove(s.submitIDs, aResp.ID.(uint64))
	}
}

func (s *Stratum) handleStratumMsg(resp interface{}) error {
	nResp := resp.(StratumMsg)
	log.Trace(nResp)
	// Too much is still handled in unmarshaler.  Need to
	// move stuff other than unmarshalling here.
	switch nResp.Method {
	case "client.show_message":
		log.Info(nResp.Params)
	case "client.reconnect":
		log.Debug("Reconnect requested")
		if err := s.applyReconnect(nResp); err != nil {
			return err
		}
		return s.reconnectUntilReady()

	case "client.get_version":
		log.Debug("get_version request received.")
		msg := StratumMsg{
			Method: nResp.Method,
			ID:     nResp.ID,
			Params: []string{"excc-gominer/" + s.cfg.Version},
		}
		m, err := json.Marshal(msg)
		if err != nil {
			log.Error(err)
			return nil
		}
		s.Lock()
		err = writeStratumMessage(s.Conn, m)
		if err != nil && s.Conn != nil {
			_ = s.Conn.Close()
		}
		s.Unlock()
		if err != nil {
			log.Error(err)
		}
	}
	return nil
}

func (s *Stratum) applyReconnect(resp StratumMsg) error {
	if len(resp.Params) < 3 {
		return errJsonType
	}
	wait, err := strconv.ParseUint(resp.Params[2], 10, 64)
	if err != nil || wait > uint64((1<<63-1)/int64(time.Second)) {
		return errJsonType
	}
	time.Sleep(time.Duration(wait) * time.Second)
	s.Lock()
	s.cfg.Pool = net.JoinHostPort(resp.Params[0], resp.Params[1])
	s.Unlock()
	return nil
}

func (s *Stratum) handleNotifyRes(resp interface{}) {
	s.Lock()
	defer s.Unlock()
	nResp := resp.(NotifyRes)
	if err := validateNotify(nResp); err != nil {
		log.Errorf("Ignoring invalid notify: %v", err)
		return
	}
	s.PoolWork.JobID = nResp.JobID
	s.PoolWork.CB1 = nResp.GenTX1
	heightHex := nResp.GenTX1[186:188] + nResp.GenTX1[184:186]
	height, err := strconv.ParseInt(heightHex, 16, 32)
	if err != nil {
		log.Debugf("failed to parse height %v", err)
		height = 0
	}

	s.PoolWork.Height = height
	s.PoolWork.CB2 = nResp.GenTX2
	s.PoolWork.Hash = nResp.Hash
	s.PoolWork.Nbits = nResp.Nbits
	s.PoolWork.Version = nResp.BlockVersion
	parsedNtime, err := strconv.ParseInt(nResp.Ntime, 16, 64)
	if err != nil {
		log.Error(err)
	}

	s.PoolWork.Ntime = nResp.Ntime
	s.PoolWork.NtimeDelta = parsedNtime - time.Now().Unix()
	s.PoolWork.Clean = nResp.CleanJobs
	s.PoolWork.NewWork = true
	select {
	case s.WorkReady <- struct{}{}:
	default:
	}
	log.Trace("notify: ", spew.Sdump(nResp))
}

func (s *Stratum) handleSubscribeReply(resp interface{}) {
	s.Lock()
	defer s.Unlock()
	nResp := resp.(*SubscribeReply)
	s.PoolWork.ExtraNonce1 = nResp.ExtraNonce1
	s.PoolWork.ExtraNonce2Length = nResp.ExtraNonce2Length
	log.Debug("Subscribe reply received.")
	log.Trace(spew.Sdump(resp))
}

func writeStratumMessage(conn net.Conn, message []byte) (err error) {
	if conn == nil {
		return errors.New("stratum connection is closed")
	}
	if err := conn.SetWriteDeadline(time.Now().Add(reconnectRetryDelay)); err != nil {
		return err
	}
	defer func() {
		if clearErr := conn.SetWriteDeadline(time.Time{}); err == nil {
			err = clearErr
		}
	}()
	message = append(message, '\n')
	n, err := conn.Write(message)
	if err == nil && n != len(message) {
		err = io.ErrShortWrite
	}
	return err
}

// Auth sends a message to the pool to authorize a worker.
func (s *Stratum) Auth() error {
	s.Lock()
	defer s.Unlock()
	return s.auth()
}

func (s *Stratum) auth() error {
	msg := StratumMsg{
		Method: "mining.authorize",
		ID:     s.ID,
		Params: []string{s.cfg.User, s.cfg.Pass},
	}
	// Auth reply has no method so need a way to identify it.
	// Ugly, but not much choice.
	id, ok := msg.ID.(uint64)
	if !ok {
		return errJsonType
	}
	s.authID = id
	s.ID++
	log.Tracef("%v", msg)
	m, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return writeStratumMessage(s.Conn, m)
}

// Subscribe sends the subscribe message to get mining info for a worker.
func (s *Stratum) Subscribe() error {
	s.Lock()
	defer s.Unlock()
	return s.subscribe()
}

func (s *Stratum) subscribe() error {
	msg := StratumMsg{
		Method: "mining.subscribe",
		ID:     s.ID,
		Params: []string{"excc-gominer/" + s.cfg.Version},
	}
	s.subID = msg.ID.(uint64)
	s.ID++
	m, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	log.Tracef("%v", string(m))
	return writeStratumMessage(s.Conn, m)
}

// Unmarshal provides a json unmarshaler for the commands.
// I'm sure a lot of this can be generalized but the json we deal with
// is pretty yucky.
func (s *Stratum) Unmarshal(blob []byte) (interface{}, error) {
	s.Lock()
	defer s.Unlock()
	var (
		objmap map[string]json.RawMessage
		method string
		id     uint64
	)

	err := json.Unmarshal(blob, &objmap)
	if err != nil {
		return nil, err
	}
	// decode command
	// Not everyone has a method.
	err = json.Unmarshal(objmap["method"], &method)
	if err != nil {
		method = ""
	}

	if _, ok := objmap["id"]; ok {
		err = json.Unmarshal(objmap["id"], &id)
		if err != nil {
			return nil, err
		}
	}

	log.Trace("Received: method: ", method, " id: ", id)
	if id == s.authID {
		return unmarshalBasicReply(objmap)
	}
	if id == s.subID {
		var resJS []json.RawMessage
		rawResult, ok := objmap["result"]
		if !ok {
			return nil, errJsonType
		}
		if bytes.Equal(rawResult, []byte("null")) || bytes.Equal(rawResult, []byte("false")) {
			return unmarshalBasicReply(objmap)
		}
		if err := json.Unmarshal(rawResult, &resJS); err != nil {
			return nil, err
		}
		if len(resJS) < 3 {
			return nil, errJsonType
		}
		var subscriptions []json.RawMessage
		if err := json.Unmarshal(resJS[0], &subscriptions); err != nil {
			return nil, err
		}
		if len(subscriptions) == 0 {
			return nil, errJsonType
		}
		resp := &SubscribeReply{}
		if err := json.Unmarshal(resJS[1], &resp.ExtraNonce1); err != nil {
			return nil, err
		}
		if err := json.Unmarshal(resJS[2], &resp.ExtraNonce2Length); err != nil {
			return nil, err
		}
		extraNonce1, err := hex.DecodeString(resp.ExtraNonce1)
		if err != nil {
			return nil, fmt.Errorf("invalid extra nonce 1: %w", err)
		}
		if len(extraNonce1) > 32 {
			return nil, errors.New("invalid extra nonce lengths")
		}
		extraNonce2Length := uint32(resp.ExtraNonce2Length)
		if resp.ExtraNonce2Length != float64(extraNonce2Length) ||
			extraNonce2Length > uint32(32-len(extraNonce1)) {
			return nil, errors.New("invalid extra nonce lengths")
		}
		return resp, nil
	}
	if sliceContains(s.submitIDs, id) {
		return unmarshalBasicReply(objmap)
	}
	switch method {
	case "mining.notify":
		log.Trace("Unmarshal mining.notify")
		var params []json.RawMessage
		if err := json.Unmarshal(objmap["params"], &params); err != nil {
			return nil, err
		}
		if len(params) < 9 {
			return nil, errJsonType
		}
		nres := NotifyRes{}
		fields := []interface{}{
			&nres.JobID,
			&nres.Hash,
			&nres.GenTX1,
			&nres.GenTX2,
			&nres.MerkleBranches,
			&nres.BlockVersion,
			&nres.Nbits,
			&nres.Ntime,
			&nres.CleanJobs,
		}
		for i := range fields {
			if err := json.Unmarshal(params[i], fields[i]); err != nil {
				return nil, err
			}
		}
		if err := validateNotify(nres); err != nil {
			return nil, err
		}
		return nres, nil

	case "mining.set_difficulty":
		log.Trace("Received new difficulty.")
		var params []json.RawMessage
		if err := json.Unmarshal(objmap["params"], &params); err != nil {
			return nil, err
		}
		if len(params) != 1 {
			return nil, errJsonType
		}
		var difficulty float64
		if err := json.Unmarshal(params[0], &difficulty); err != nil {
			return nil, err
		}
		target, err := util.DiffToTarget(difficulty, chainParams.PowLimit)
		if err != nil {
			return nil, err
		}
		s.Target = target
		s.Diff = difficulty
		var nres = StratumMsg{}
		nres.Method = method
		diffStr := strconv.FormatFloat(difficulty, 'E', -1, 32)
		nres.Params = []string{diffStr}
		log.Infof("Stratum difficulty set to %v", difficulty)
		return nres, nil

	case "client.show_message":
		rawParams, ok := objmap["params"]
		if !ok {
			rawParams = objmap["result"]
		}
		var params []string
		if err := json.Unmarshal(rawParams, &params); err != nil {
			return nil, err
		}
		if len(params) != 1 {
			return nil, errJsonType
		}
		return StratumMsg{Method: method, Params: params}, nil

	case "client.get_version":
		var nres = StratumMsg{}
		var id uint64
		err = json.Unmarshal(objmap["id"], &id)
		if err != nil {
			return nil, err
		}
		nres.Method = method
		nres.ID = id
		return nres, nil

	case "client.reconnect":
		var nres StratumMsg
		var id uint64
		if err := json.Unmarshal(objmap["id"], &id); err != nil {
			return nil, err
		}
		nres.Method = method
		nres.ID = id

		var params []json.RawMessage
		if err := json.Unmarshal(objmap["params"], &params); err != nil {
			return nil, err
		}
		if len(params) < 3 {
			return nil, errJsonType
		}
		var hostname string
		if err := json.Unmarshal(params[0], &hostname); err != nil {
			return nil, err
		}
		var port, wait uint64
		if err := json.Unmarshal(params[1], &port); err != nil {
			return nil, err
		}
		if port == 0 || port > 65535 {
			return nil, errJsonType
		}
		if err := json.Unmarshal(params[2], &wait); err != nil {
			return nil, err
		}
		nres.Params = []string{hostname, strconv.FormatUint(port, 10), strconv.FormatUint(wait, 10)}

		return nres, nil

	default:
		resp := &StratumRsp{}
		err := json.Unmarshal(blob, &resp)
		if err != nil {
			return nil, err
		}
		return resp, nil
	}
}

// PrepWork converts the stratum notify to getwork style data for mining.
func (s *Stratum) PrepWork() error {
	// Build final extranonce, which is basically the pool user and worker ID.
	extraNonce, err := hex.DecodeString(s.PoolWork.ExtraNonce1)
	if err != nil {
		log.Error("Error decoding ExtraNonce1.")
		return err
	}

	cb1, err := hex.DecodeString(s.PoolWork.CB1)
	if err != nil {
		log.Error("Error decoding Coinbase pt 1.")
		return err
	}
	if len(cb1) < minCoinbase1Size {
		return fmt.Errorf("coinbase part 1 is %d bytes, need at least %d", len(cb1), minCoinbase1Size)
	}

	cb2, err := hex.DecodeString(s.PoolWork.CB2)
	if err != nil {
		log.Error("Error decoding Coinbase pt 2.")
		return err
	}

	v, err := util.ReverseToInt(s.PoolWork.Version)
	if err != nil {
		return err
	}
	version := new(bytes.Buffer)
	err = binary.Write(version, binary.LittleEndian, v)
	if err != nil {
		return err
	}
	prevHash, err := hex.DecodeString(s.PoolWork.Hash)
	if err != nil {
		log.Error("Error encoding previous hash.")
		return err
	}

	var workdata [work.GetworkDataLen]byte
	workPosition := 0
	copy(workdata[workPosition:], version.Bytes())
	workPosition += 4
	copy(workdata[workPosition:], prevHash)
	workPosition += 32
	copy(workdata[workPosition:], cb1[0:108])
	workPosition = 144
	copy(workdata[workPosition:], extraNonce)
	workPosition = 176
	copy(workdata[workPosition:], cb2[:])

	bh := wire.BlockHeader{}
	bh.FromBytes(workdata[:])

	givenTs := binary.LittleEndian.Uint32(workdata[128+4*work.TimestampWord : 132+4*work.TimestampWord])
	atomic.StoreUint32(&s.latestJobTime, givenTs)

	if s.Target == nil {
		return errors.New("no target set")
	}

	w := work.NewWork(bh, s.Target, givenTs, uint32(time.Now().Unix()), false, s.PoolWork.JobID)
	s.PoolWork.Work = w

	return nil
}

// PrepSubmit formats a mining.sumbit message from the solved work.
func (s *Stratum) PrepSubmit(data []byte, jobID string) (Submit, error) {
	log.Debugf("Stratum got valid work to submit %x", data)

	sub := Submit{}
	sub.Method = "mining.submit"

	// Format data to send off.
	hexData := hex.EncodeToString(data)
	decodedData, err := hex.DecodeString(hexData)
	if err != nil {
		log.Error("Error decoding data")
		return sub, err
	}

	var submittedHeader wire.BlockHeader
	bhBuf := bytes.NewReader(decodedData[0:wire.MaxBlockHeaderPayload])
	err = submittedHeader.Deserialize(bhBuf)
	if err != nil {
		log.Error("Error generating header")
		return sub, err
	}

	latestWorkTs := atomic.LoadUint32(&s.latestJobTime)
	if uint32(submittedHeader.Timestamp.Unix()) != latestWorkTs {
		return sub, ErrStratumStaleWork
	}

	sub.ID = s.ID
	s.submitIDs = append(s.submitIDs, s.ID)
	s.ID++

	// The timestamp string should be:
	//
	//   timestampStr := fmt.Sprintf("%08x",
	//     uint32(submittedHeader.Timestamp.Unix()))
	//
	// but the "stratum" protocol appears to only use this value
	// to check if the miner is in sync with the latest announcement
	// of work from the pool. If this value is anything other than
	// the timestamp of the latest pool work timestamp, work gets
	// rejected from the current implementation.
	timestampStr := fmt.Sprintf("%08x", latestWorkTs)
	xnonceStr := hex.EncodeToString(data[144:156])
	nonceStr := hex.EncodeToString(data[140:144])
	solutionStr := hex.EncodeToString(submittedHeader.EquihashSolution[:])

	sub.Params = []string{s.cfg.User, jobID, xnonceStr, timestampStr, nonceStr, solutionStr}

	return sub, nil
}
