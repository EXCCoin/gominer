# gominer

gominer mines ExchangeCoin (EXCC) using Equihash 144,5. It supports solo and
Stratum mining.

| Backend | GPUs | Platform |
| --- | --- | --- |
| CUDA | NVIDIA | Linux amd64 |
| HIP | AMD | Linux amd64 |
| wgpu | AMD, NVIDIA, Intel, Apple | Linux, Windows, macOS |

CUDA is the fastest option on NVIDIA. HIP is the native AMD option. wgpu is
the portable option and uses Vulkan, DirectX 12, or Metal.

## Build

Use the Go version declared in `go.mod`.

CUDA requires CUDA 12.8 or newer:

```sh
CUDA_HOME=/path/to/cuda ./build.sh
CUDA_HOME=/path/to/cuda ./build.sh test
```

HIP requires ROCm HIP:

```sh
./build-hip.sh
./build-hip.sh test
```

wgpu requires Rust:

```sh
./build-wgpu.sh
```

On Windows amd64, install Go, Rust, and MinGW-w64 GCC, then run:

```powershell
./build-wgpu.ps1
```

Keep `eqwgpu1445.dll` beside `gominer-wgpu.exe`. GitHub Actions builds and
tests the wgpu variant on Linux and Windows amd64; CUDA and HIP solution tests
still require matching GPU hardware.

## Run

Benchmark:

```sh
./gominer-wgpu -B -I 1
```

Solo mining against exccd:

```sh
./gominer -u rpcusername -P rpcpassword
```

Stratum mining:

```sh
./gominer -o stratum+tcp://pool:port -m username -n password
```

Use `-l` to list GPUs, `-D` to select GPUs, `-I` to set solver instances, and
`-W` to set work size. Start with one instance and benchmark before increasing
it. See [sample-gominer.conf](sample-gominer.conf) for the remaining options.

## Status API

`--apilisten=localhost` exposes JSON status at `http://localhost:3333/`.

## Tested performance

These are measured results, not estimates. Performance varies with drivers,
power limits, and cooling.

| GPU | Backend | Commit | Settings | Result |
| --- | --- | --- | --- | ---: |
| Radeon 8060S | wgpu | `2db0a8b` | `-B -I 1 -W 1048576`, 90 s | 45.3 Sol/s |
| Radeon 8060S | HIP | `2db0a8b` | `-B -I 1 -W 5592320`, 90 s | 61.3 Sol/s |
| RTX 3050 Ti Laptop | CUDA | `5620d62` | `-B -I 1`, 30 s | 18.0 Sol/s |
| RTX 4090 | CUDA | `5620d62` | `-B -I 4`, 30 s | 208.9 Sol/s |
| RTX 5090 | CUDA | `5620d62` | `-B -I 4`, 60 s | 299.4 Sol/s |

## License

[GPL-3.0-only](LICENSE)
