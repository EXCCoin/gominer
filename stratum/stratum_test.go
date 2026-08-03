package stratum

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/EXCCoin/gominer/work"
)

func subscribeReply(id uint64) string {
	return fmt.Sprintf(`{"id":%d,"result":[[["mining.notify","session"]],"00",4],"error":null}`, id)
}

func notifyMessage() string {
	return `{"id":null,"method":"mining.notify","params":["job","` +
		strings.Repeat("0", 64) + `","` + strings.Repeat("0", minCoinbase1Size*2) +
		`","",[],"01000000","1a12334a","00000000",true]}`
}

func writeLine(conn net.Conn, line string) {
	_, _ = conn.Write([]byte(line + "\n"))
}

type deadlineObserverConn struct {
	net.Conn
	extended   chan struct{}
	cleared    chan struct{}
	extendOnce sync.Once
	clearOnce  sync.Once
}

type shortWriteConn struct {
	net.Conn
}

func (c shortWriteConn) Write(message []byte) (int, error) {
	return len(message) - 1, nil
}

func (c *deadlineObserverConn) SetReadDeadline(deadline time.Time) error {
	if deadline.IsZero() {
		c.clearOnce.Do(func() { close(c.cleared) })
	} else if time.Until(deadline) > 2*reconnectRetryDelay {
		c.extendOnce.Do(func() { close(c.extended) })
	}
	return c.Conn.SetReadDeadline(deadline)
}

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
	n := NotifyRes{
		JobID:        "1",
		Hash:         strings.Repeat("0", 64),
		GenTX1:       strings.Repeat("0", minCoinbase1Size*2),
		BlockVersion: "01000000",
		Nbits:        "1a12334a",
		Ntime:        "00000000",
	}

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

func TestMalformedMessagesReturnErrors(t *testing.T) {
	tests := []struct {
		name  string
		blob  string
		subID uint64
	}{
		{"notify", `{"method":"mining.notify","params":["job"]}`, 0},
		{"subscribe", `{"id":1,"result":[[],"00",4]}`, 1},
		{"subscribe extra nonce", `{"id":1,"result":[[["mining.notify","session"]],"zz",4]}`, 1},
		{"difficulty", `{"method":"mining.set_difficulty","params":[]}`, 0},
		{"show message", `{"method":"client.show_message","result":[]}`, 0},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			subID := test.subID
			if subID == 0 {
				subID = 98
			}
			s := &Stratum{subID: subID, authID: 99}
			defer func() {
				if recovered := recover(); recovered != nil {
					t.Errorf("Unmarshal panicked: %v", recovered)
				}
			}()
			if _, err := s.Unmarshal([]byte(test.blob)); err == nil {
				t.Fatal("malformed message was accepted")
			}
		})
	}
}

func TestMalformedNotifyDoesNotReplaceWork(t *testing.T) {
	s := &Stratum{WorkReady: make(chan struct{}, 1)}
	s.handleNotifyRes(NotifyRes{GenTX1: "00", Ntime: "00000000"})
	if s.PoolWork.NewWork {
		t.Fatal("malformed notify replaced pool work")
	}
}

func TestPrepWorkRejectsShortCoinbase(t *testing.T) {
	s := &Stratum{Target: big.NewInt(1)}
	s.PoolWork = NotifyWork{
		CB1:         strings.Repeat("0", 188),
		Hash:        strings.Repeat("0", 64),
		Version:     "01000000",
		ExtraNonce1: "",
	}
	if err := s.PrepWork(); err == nil {
		t.Fatal("short coinbase was accepted")
	}
}

func TestArrayErrorRejectsShareAndClearsID(t *testing.T) {
	for _, reply := range []string{
		`{"result":null,"error":[21,"Job not found",null],"id":4}`,
		`{"error":[21,"Job not found",null],"id":4}`,
	} {
		s := &Stratum{authID: 99, submitIDs: []uint64{4}}
		resp, err := s.Unmarshal([]byte(reply))
		if err != nil {
			t.Fatal(err)
		}
		s.handleBasicReply(resp)
		if got := atomic.LoadUint64(&s.InvalidShares); got != 1 {
			t.Fatalf("invalid shares = %d, want 1", got)
		}
		if len(s.submitIDs) != 0 {
			t.Fatalf("submit IDs were not cleared: %v", s.submitIDs)
		}
	}
}

func TestInvalidDifficultyPreservesTarget(t *testing.T) {
	target := big.NewInt(1)
	s := &Stratum{authID: 98, subID: 99, Diff: 1, Target: target}
	if _, err := s.Unmarshal([]byte(`{"method":"mining.set_difficulty","params":[0]}`)); err == nil {
		t.Fatal("nonpositive difficulty was accepted")
	}
	if s.Target != target || s.Diff != 1 {
		t.Fatal("invalid difficulty changed active target")
	}
}

func TestPrepSubmitAdvancesPastReservedID(t *testing.T) {
	s := &Stratum{ID: 7}
	sub, err := s.PrepSubmit(make([]byte, work.GetworkDataLen), "job")
	if err != nil {
		t.Fatal(err)
	}
	if sub.ID != uint64(7) || s.ID != 8 {
		t.Fatalf("submit ID = %v, next ID = %d", sub.ID, s.ID)
	}
}

func TestShowMessageUsesParams(t *testing.T) {
	s := &Stratum{authID: 98, subID: 99}
	resp, err := s.Unmarshal([]byte(`{"method":"client.show_message","params":["maintenance"]}`))
	if err != nil {
		t.Fatal(err)
	}
	msg := resp.(StratumMsg)
	if len(msg.Params) != 1 || msg.Params[0] != "maintenance" {
		t.Fatalf("message params = %v", msg.Params)
	}
}

func TestReadStratumMessageCapsLineLength(t *testing.T) {
	reader := bufio.NewReaderSize(strings.NewReader(strings.Repeat("x", maxStratumMessageSize+1)), maxStratumMessageSize)
	if _, err := readStratumMessage(reader); !errors.Is(err, errStratumMessageTooLong) {
		t.Fatalf("error = %v, want %v", err, errStratumMessageTooLong)
	}
}

func TestWriteStratumMessageRejectsShortWrite(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()
	if err := writeStratumMessage(shortWriteConn{client}, []byte("{}")); !errors.Is(err, io.ErrShortWrite) {
		t.Fatalf("error = %v, want %v", err, io.ErrShortWrite)
	}
}

func TestHandshakeBoundsWaitForUsableNotify(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()
	observed := &deadlineObserverConn{
		Conn:     client,
		extended: make(chan struct{}),
		cleared:  make(chan struct{}),
	}
	s := &Stratum{
		Conn:      observed,
		Reader:    bufio.NewReaderSize(observed, maxStratumMessageSize),
		ID:        1,
		WorkReady: make(chan struct{}, 1),
	}
	s.Target = big.NewInt(1)
	sendNotify := make(chan struct{})
	go func() {
		scanner := bufio.NewScanner(server)
		for i := 0; i < 2; i++ {
			if !scanner.Scan() {
				return
			}
		}
		writeLine(server, subscribeReply(1))
		writeLine(server, `{"id":2,"result":true,"error":null}`)
		writeLine(server, strings.Replace(notifyMessage(), strings.Repeat("0", minCoinbase1Size*2), strings.Repeat("g", minCoinbase1Size*2), 1))
		<-sendNotify
		writeLine(server, notifyMessage())
	}()

	done := make(chan error, 1)
	go func() { done <- s.handshake(observed) }()
	select {
	case <-observed.extended:
	case <-time.After(time.Second):
		t.Fatal("first-work deadline was not extended after subscribe and authorize")
	}
	select {
	case err := <-done:
		t.Fatalf("handshake accepted unusable notify: %v", err)
	default:
	}
	close(sendNotify)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("handshake did not accept delayed notify")
	}
	select {
	case <-observed.cleared:
	default:
		t.Fatal("read deadline was not cleared after usable notify")
	}
}

func TestHandshakeDoesNotConfuseSubmitReplyWithAuthorization(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()
	s := &Stratum{
		Conn:      client,
		Reader:    bufio.NewReaderSize(client, maxStratumMessageSize),
		ID:        1,
		submitIDs: []uint64{99},
		WorkReady: make(chan struct{}, 1),
	}
	sendAuth := make(chan struct{})
	earlyRepliesSent := make(chan struct{})
	go func() {
		scanner := bufio.NewScanner(server)
		for i := 0; i < 2; i++ {
			if !scanner.Scan() {
				return
			}
		}
		writeLine(server, subscribeReply(1))
		writeLine(server, `{"id":99,"result":true,"error":null}`)
		writeLine(server, notifyMessage())
		close(earlyRepliesSent)
		<-sendAuth
		writeLine(server, `{"id":2,"result":true,"error":null}`)
	}()

	done := make(chan error, 1)
	go func() { done <- s.handshake(client) }()
	select {
	case <-earlyRepliesSent:
	case <-time.After(time.Second):
		t.Fatal("timed out sending early replies")
	}
	select {
	case err := <-done:
		t.Fatalf("submit reply authorized handshake: %v", err)
	case <-time.After(50 * time.Millisecond):
	}
	close(sendAuth)
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("handshake did not accept authorization reply")
	}
	if got := atomic.LoadUint64(&s.ValidShares); got != 1 || len(s.submitIDs) != 0 {
		t.Fatalf("submit reply not handled: valid=%d ids=%v", got, s.submitIDs)
	}
}

func TestHandshakeSerializesSubscribeAndAuthorize(t *testing.T) {
	client, server := net.Pipe()
	defer client.Close()
	defer server.Close()
	s := &Stratum{
		Conn:      client,
		Reader:    bufio.NewReaderSize(client, maxStratumMessageSize),
		ID:        1,
		WorkReady: make(chan struct{}, 1),
	}
	done := make(chan error, 1)
	go func() { done <- s.handshake(client) }()
	scanner := bufio.NewScanner(server)
	if !scanner.Scan() {
		t.Fatal("subscribe was not written")
	}
	lockAcquired := make(chan struct{})
	go func() {
		s.Lock()
		s.Unlock()
		close(lockAcquired)
	}()
	select {
	case <-lockAcquired:
		t.Fatal("handshake lock released between subscribe and authorize")
	case <-time.After(50 * time.Millisecond):
	}
	if !scanner.Scan() {
		t.Fatal("authorize was not written")
	}
	select {
	case <-lockAcquired:
	case <-time.After(time.Second):
		t.Fatal("handshake lock was not released after authorize")
	}
	writeLine(server, subscribeReply(1))
	writeLine(server, `{"id":2,"result":true,"error":null}`)
	writeLine(server, notifyMessage())
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("handshake did not complete")
	}
}

func TestStratumConnReturnsAuthorizationRejection(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		scanner := bufio.NewScanner(conn)
		for i := 0; i < 2; i++ {
			if !scanner.Scan() {
				return
			}
		}
		writeLine(conn, subscribeReply(1))
		writeLine(conn, `{"id":2,"result":null,"error":[24,"Unauthorized worker",null]}`)
	}()

	s, err := StratumConn("stratum+tcp://"+listener.Addr().String(), "bad", "bad", "", "", "", "test")
	if s != nil {
		_ = s.Conn.Close()
		t.Fatal("connection succeeded with rejected credentials")
	}
	if !errors.Is(err, errAuthorizationRejected) {
		t.Fatalf("error = %v, want authorization rejection", err)
	}
}

func TestReconnectStopsRetryingRejectedCredentials(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer listener.Close()
	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		scanner := bufio.NewScanner(conn)
		for i := 0; i < 2; i++ {
			if !scanner.Scan() {
				return
			}
		}
		writeLine(conn, subscribeReply(1))
		writeLine(conn, `{"id":2,"result":false,"error":[24,"Unauthorized worker",null]}`)
	}()

	s := &Stratum{ID: 1, WorkReady: make(chan struct{}, 1), Errors: make(chan error, 1)}
	s.cfg = Config{Pool: listener.Addr().String(), User: "bad", Pass: "bad", Version: "test"}
	done := make(chan error, 1)
	go func() { done <- s.reconnectUntilReady() }()
	select {
	case err := <-done:
		if !errors.Is(err, errAuthorizationRejected) {
			t.Fatalf("error = %v, want authorization rejection", err)
		}
	case <-time.After(time.Second):
		t.Fatal("reconnect kept retrying rejected credentials")
	}
}

func TestReconnectRequestDuringHandshakeUsesNewConnection(t *testing.T) {
	first, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer first.Close()
	second, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer second.Close()
	host, portString, err := net.SplitHostPort(second.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	port, err := strconv.Atoi(portString)
	if err != nil {
		t.Fatal(err)
	}
	releaseSecond := make(chan struct{})
	go func() {
		conn, err := first.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		scanner := bufio.NewScanner(conn)
		for i := 0; i < 2; i++ {
			if !scanner.Scan() {
				return
			}
		}
		writeLine(conn, fmt.Sprintf(`{"id":0,"method":"client.reconnect","params":[%q,%d,0]}`, host, port))
	}()
	go func() {
		conn, err := second.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		scanner := bufio.NewScanner(conn)
		for i := 0; i < 2; i++ {
			if !scanner.Scan() {
				return
			}
		}
		writeLine(conn, subscribeReply(3))
		writeLine(conn, `{"id":4,"result":true,"error":null}`)
		writeLine(conn, notifyMessage())
		<-releaseSecond
	}()

	s := &Stratum{ID: 1, WorkReady: make(chan struct{}, 1), Errors: make(chan error, 1)}
	s.cfg = Config{Pool: first.Addr().String(), User: "user", Pass: "pass", Version: "test"}
	done := make(chan error, 1)
	go func() { done <- s.Reconnect() }()
	select {
	case err := <-done:
		close(releaseSecond)
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		close(releaseSecond)
		t.Fatal("reconnect nested or stalled after handshake redirect")
	}
	defer s.Conn.Close()
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
			strings.Repeat("0", 64) + `","` + strings.Repeat("0", minCoinbase1Size*2) +
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
