package nvml

/*
#cgo !windows LDFLAGS: -L/opt/cuda/lib64 -L/opt/cuda/lib -L/usr/local/cuda/lib64 -L/usr/lib/nvidia-396 -lnvidia-ml
#cgo windows LDFLAGS: -L../nvidia/NVSMI/ -lnvml
*/
import "C"
