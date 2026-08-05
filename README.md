# gominer

gominer mines ExchangeCoin (EXCC) with Equihash 144,5. It supports solo
mining against exccd and Stratum pool mining.

## GPU backends

| Backend | Hardware | Platform | Binary |
| --- | --- | --- | --- |
| CUDA | NVIDIA | Linux amd64 | `gominer` |
| HIP | AMD | Linux amd64 | `gominer-hip` |
| wgpu | AMD, NVIDIA, Intel, Apple | Linux, Windows, macOS | `gominer-wgpu` |

CUDA is the native Linux backend for NVIDIA, and HIP is the native Linux
backend for AMD. wgpu is the portable option and uses Vulkan on Linux,
DirectX 12 on Windows, and Metal on macOS.

## Downloads

The `v1.1.1-beta` release provides portable wgpu builds for amd64:

- [Linux amd64](https://github.com/EXCCoin/gominer/releases/download/v1.1.1-beta/gominer-v1.1.1-beta-linux-amd64.tar.gz)
- [Windows amd64](https://github.com/EXCCoin/gominer/releases/download/v1.1.1-beta/gominer-v1.1.1-beta-windows-amd64.zip)
- [SHA256 checksums](https://github.com/EXCCoin/gominer/releases/download/v1.1.1-beta/SHA256SUMS)

The Windows archive includes all required DLLs. CUDA and HIP
binaries depend on the target GPU/toolchain and are built from source.

## Building

All builds require the Go version declared in [`go.mod`](go.mod).

### CUDA

The CUDA build requires Linux amd64, an NVIDIA driver, and CUDA 12.8 or newer.

```sh
CUDA_HOME=/path/to/cuda ./build.sh
CUDA_HOME=/path/to/cuda ./build.sh test
```

The test build generates solutions on every available GPU and verifies them
on the CPU. The default binary contains code for Turing through Blackwell.
Pascal and Volta need a CUDA 12.x toolkit and a custom `GENCODE` value; see
[`build.sh`](build.sh).

### HIP

The HIP build requires Linux amd64 and ROCm HIP.

```sh
./build-hip.sh
./build-hip.sh test
```

`GPU_ARCH` can be set when the build host cannot detect the target GPU:

```sh
GPU_ARCH=gfx1151 ./build-hip.sh
```

### wgpu

Linux and macOS builds require Rust:

```sh
./build-wgpu.sh
```

Running the binary requires Vulkan on Linux or Metal on macOS.

Linux arm64 cross-compilation also needs the AArch64 GCC/G++ cross compiler:

```sh
rustup target add aarch64-unknown-linux-gnu
CARGO_BUILD_TARGET=aarch64-unknown-linux-gnu ./build-wgpu.sh
```

Windows amd64 requires Go, Rust, and MinGW-w64 GCC:

```powershell
./build-wgpu.ps1
```

Keep the generated DLLs beside `gominer-wgpu.exe`. The
[`CI` workflow](.github/workflows/ci.yml) builds and tests wgpu on Linux and
Windows amd64. CUDA and HIP tests run separately on matching hardware.

## Running

Use the binary for the backend you built. The examples below use the CUDA
binary.

List GPUs and run a benchmark:

```sh
./gominer -l
./gominer -B -I 1
```

Solo mining against a local exccd:

```sh
./gominer -u rpcusername -P rpcpassword
```

The default mainnet RPC address is `localhost:9109`. TLS is enabled and
exccd's default `rpc.cert` is used unless `--rpccert` selects another CA file.

Stratum mining:

```sh
./gominer -o stratum+tcp://pool:port -m username -n password
```

Useful mining options:

- `-D 0,1` selects devices.
- `-I 1` sets concurrent solver instances per device. A native solver instance
  uses about 3 GB of GPU memory.
- `-W 1048576` sets the solver work size. Values must be multiples of 256.

Start with one instance and benchmark before increasing it. Command-line
options can also be placed in [`sample-gominer.conf`](sample-gominer.conf);
`gominer -h` shows the default configuration path and all available options.

## Status API

`--apilisten` enables the HTTP status endpoint. If no port is given, it uses
port `3333`.

```sh
./gominer-wgpu -B -I 1 --apilisten=127.0.0.1:3333
curl http://127.0.0.1:3333/
```

Example benchmark response:

```json
{
  "validShares": 0,
  "staleShares": 0,
  "invalidShares": 0,
  "totalShares": 0,
  "sharesPerMinute": 0,
  "started": 1785833899,
  "uptime": 4,
  "devices": [
    {
      "index": 0,
      "deviceName": "NVIDIA GeForce RTX 5090 (Vulkan)",
      "deviceType": "GPU",
      "hashRate": 236.25,
      "hashRateFormatted": "236.2 Sol/s",
      "fanPercent": 0,
      "temperature": 0,
      "started": 1785833899
    }
  ]
}
```

Pool mining responses also include `pool.started` and `pool.uptime`. The API
has no authentication or TLS, so keep it on loopback unless access is
restricted elsewhere.

## Tested hardware

Recorded rates are in Equihash solutions per second (Sol/s).

| Date | Host and GPU | Backend | Commit and settings | Result |
| --- | --- | --- | --- | ---: |
| 2026-07-30 | Apple M4 Max, 16-core CPU, 40-core GPU, 128 GB | wgpu 22.1, Metal, macOS 26.4.1 | `718f456`, `-I 1 -W 1048576` | 52–55 Sol/s |
| 2026-07-31 | Ryzen AI MAX+ 395, Radeon 8060S, 62.4 GiB | wgpu 22.1, RADV 25.3.6, Fedora 43 | `2db0a8b`, `-B -I 1 -W 1048576`, 90 s | 45.3 Sol/s |
| 2026-07-31 | Ryzen AI MAX+ 395, Radeon 8060S, 62.4 GiB | HIP, ROCm 6.4.2, Fedora 43 | `2db0a8b`, `-B -I 1 -W 5592320`, 90 s | 61.3 Sol/s |
| 2026-07-30 | Core i9-13900HK, RTX 3050 Ti Laptop, 32 GiB, 35 W | CUDA sm86, driver 595.71.05, Gentoo | `5620d62`, `-B -I 1`, 30 s | 18.0 Sol/s |
| 2026-07-30 | EPYC 9654, RTX 4090, 1 TiB | CUDA sm89, driver 595.71.05, Gentoo | `5620d62`, `-B -I 4`, 30 s | 208.9 Sol/s |
| 2026-07-30 | Ryzen 9 9950X3D, RTX 5090, 128 GiB | CUDA 13.3 sm120, driver 610.43.03, Fedora 44 | `5620d62`, `-B -I 4`, 60 s | 299.4 Sol/s |

The CUDA and HIP runs passed generated-solution verification. The wgpu GPU
solver test passed on the Radeon host.

## License

[GPL-3.0-only](LICENSE)
