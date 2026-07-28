//go:build !windows

package main

import (
	"os"
	"os/exec"
	"testing"
)

func TestRestartProcess(t *testing.T) {
	const marker = "GOMINER_TEST_RESTART"
	switch os.Getenv(marker) {
	case "":
		cmd := exec.Command(os.Args[0], "-test.run=^TestRestartProcess$")
		cmd.Env = append(os.Environ(), marker+"=before")
		if output, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("restart helper failed: %v\n%s", err, output)
		}
	case "before":
		t.Setenv(marker, "after")
		if err := restartProcess(); err != nil {
			t.Fatal(err)
		}
		t.Fatal("restartProcess returned after a successful exec")
	case "after":
		return
	}
}
