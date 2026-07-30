
# gominer
gominer is an application for performing Proof-of-Work (PoW) mining on the Exchange Coin network (Equihash 144,5). It supports solo and stratum/pool mining with three GPU backends:

- **CUDA** (`./build.sh`) — NVIDIA only, fastest.
- **HIP** (`./build-hip.sh`) — native AMD/ROCm solver.
- **wgpu** (`./build-wgpu.sh`) — portable: AMD, NVIDIA, and Intel GPUs via
  Vulkan (Metal/DX12 also supported by the underlying library). No CUDA
  toolkit or vendor SDK needed — just a Vulkan-capable driver at runtime.
  Prefer the optimized CUDA solver on NVIDIA.

## Downloading
Linux and Windows 64-bit binaries may be downloaded from [https://github.com/EXCCoin/excc-binaries/releases/latest](https://github.com/EXCCoin/excc-binaries/releases/latest)

## Running
Benchmark mode:
```
gominer -B
```

Solo mining on mainnet using exccd running on the local host:
```
gominer -u rpcusername -P rpcpassword
```

Stratum/pool mining:
```
gominer -o stratum+tcp://pool:port -m username -n password
```

## Status API
There is a built-in status API to report miner information. You can set an address and port with `--apilisten`. There are configuration examples on [sample-gominer.conf](sample-gominer.conf). If no port is specified, then it will listen by default on `3333`.

Example usage:
```sh
$ gominer --apilisten="localhost"
```

Example output:
```sh
$ curl http://localhost:3333/
> {
    "validShares": 0,
    "staleShares": 0,
    "invalidShares": 0,
    "totalShares": 0,
    "sharesPerMinute": 0,
    "started": 1504453881,
    "uptime": 6,
    "devices": [{
        "index": 2,
        "deviceName": "GeForce GT 750M",
        "deviceType": "GPU",
        "hashRate": 110127366.53846154,
        "hashRateFormatted": "110MH/s",
        "fanPercent": 0,
        "temperature": 0,
        "started": 1504453880
    }],
    "pool": {
        "started": 1504453881,
        "uptime": 6
    }
}
```

## Building on Linux
#### Pre-Requisites
- The Go version declared in `go.mod` (automatic toolchain download is supported)
- A recent NVIDIA driver
- CUDA toolkit >= 12.8 from [here](https://developer.nvidia.com/cuda-downloads)
  * CUDA 13.x covers Turing (GTX 16xx / RTX 20xx) through Blackwell (RTX 50xx).
    For Pascal (GTX 10xx) build with a CUDA 12.x toolkit and override `GENCODE`
    (see `build.sh`).
  * A user-space install works fine, no root needed:
    `./cuda_*_linux.run --nox11 --silent --toolkit --toolkitpath=$HOME/cuda`

#### Instructions
```
git clone https://github.com/EXCCoin/gominer
cd gominer
CUDA_HOME=/path/to/cuda ./build.sh        # build
CUDA_HOME=/path/to/cuda ./build.sh test   # build + verify GPU solutions on the CPU
```

## Building the native AMD variant
```
# needs ROCm HIP, no CUDA:
./build-hip.sh          # produces ./gominer-hip
./build-hip.sh test     # build + verify GPU solutions on the CPU

# Override detection when building without direct GPU access:
GPU_ARCH=gfx1151 ./build-hip.sh
```

## Building the portable (AMD/NVIDIA/Intel) variant
```
# needs Rust (https://rustup.rs), no CUDA:
./build-wgpu.sh          # produces ./gominer-wgpu
cd eqwgpu1445 && cargo test --release   # optional: GPU solver self-check

# Cross-compile for Linux AArch64 (needs gcc/g++-aarch64-linux-gnu):
rustup target add aarch64-unknown-linux-gnu
CARGO_BUILD_TARGET=aarch64-unknown-linux-gnu ./build-wgpu.sh
```

On a native AArch64 Linux host, use `./build-wgpu.sh` normally. At runtime the
portable backend needs a Vulkan driver that exposes the GPU with 1024 compute
invocations and 64 KiB workgroup storage; ROCm and CUDA are not required. Start
a new AMD adapter with `-I 1`, then benchmark before raising the instance count.

## Tuning
- `-I/--instances N` — concurrent solver instances per GPU, ~3GB GPU memory
  each for the native solver (CUDA default scales with VRAM, up to 4; HIP and
  wgpu default to 1).
- `-W/--worksize N` — solver thread count per instance (default 2^20).

Reference: an RTX 5090 does ~248 Sol/s with the CUDA solver defaults. The wgpu
solver reaches ~47.1 Sol/s on a Ryzen AI MAX+ 395 / Radeon 8060S with RADV and
`-I 1`. On that AMD APU, the native HIP solver is tuned with `-I 1 -W 5592320`.
Rates are reported in Sol/s (Equihash solutions per second), the unit pools use.
