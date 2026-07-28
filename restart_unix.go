//go:build !windows

package main

import (
	"os"
	"syscall"
)

func restartProcess() error {
	executable, err := os.Executable()
	if err != nil {
		return err
	}
	return syscall.Exec(executable, os.Args, os.Environ())
}
