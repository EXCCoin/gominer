
# gominer
gominer is an application for performing Proof-of-Work (PoW) mining on the Exchange Coin network (Equihash 144,5). It supports solo and stratum/pool mining with three GPU backends:

- **CUDA** (`./build.sh`) — NVIDIA only, fastest.
- **HIP** (`./build-hip.sh`) — native AMD/ROCm solver.
- **wgpu** (`./build-wgpu.sh`) — portable: AMD, NVIDIA, and Intel GPUs via
  Vulkan, and Apple GPUs through native Metal. No CUDA toolkit, MoltenVK, or
  vendor SDK is needed.
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

`build.sh` produces one `./gominer` fat binary. It contains native CUDA code
for Turing through Blackwell and selects the tuned sm86 or modern solver path
at runtime. The unified build at commit `5620d62` verified 42/42 generated
solutions on RTX 3050 Ti, RTX 4090, and RTX 5090 GPUs.

## Building the native AMD variant
```
# needs ROCm HIP, no CUDA:
./build-hip.sh          # produces ./gominer-hip
./build-hip.sh test     # build + verify GPU solutions on the CPU

# Override detection when building without direct GPU access:
GPU_ARCH=gfx1151 ./build-hip.sh
```

## Building the portable GPU variant
```
# needs Rust (https://rustup.rs), no CUDA:
./build-wgpu.sh          # produces ./gominer-wgpu
cd eqwgpu1445 && cargo test --release   # optional: GPU solver self-check

# Cross-compile for Linux AArch64 (needs gcc/g++-aarch64-linux-gnu):
rustup target add aarch64-unknown-linux-gnu
CARGO_BUILD_TARGET=aarch64-unknown-linux-gnu ./build-wgpu.sh
```

On a native AArch64 Linux host, use `./build-wgpu.sh` normally. At runtime the
portable backend needs a Vulkan driver; ROCm and CUDA are not required. It
automatically selects a 64, 32, or 16 KiB collision kernel based on the GPU's
compute limits. Start a new adapter with `-I 1`, then benchmark before raising
the instance count.

## Building on macOS (Apple silicon)

The portable backend uses Metal directly through wgpu. MoltenVK and a Vulkan
SDK are not required.

```sh
xcode-select --install             # if the command-line tools are absent
brew install go rust
./build-wgpu.sh                    # produces ./gominer-wgpu
./gominer-wgpu -l                  # should list "Apple ... (Metal)"
./gominer-wgpu -B -I 1             # benchmark before connecting to a pool
```

The Metal path selects a collision kernel that fits Apple's 32 KiB threadgroup
memory limit while preserving the full Equihash collision key. On the tested
40-core M4 Max, the dedicated final-round collision table improved the same
100-nonce solver benchmark from 53.238 to 54.665 Sol/s (+2.7%). Sustained miner
runs measure about 52–55 Sol/s with the default work size and one instance; two
instances are slower. The wgpu backend cannot currently report Apple GPU
temperature or control its fans, so sustained laptop performance can vary with
cooling and power mode. For repeatable mining benchmarks, connect AC power and
select Automatic or High Power under System Settings > Battery > Energy Mode;
Low Power Mode intentionally reduces sustained performance.

## Tuning

- `-I/--instances N` — concurrent solver instances per GPU, ~3GB GPU memory
  each for the native solver (CUDA default scales with VRAM, up to 4; HIP and
  wgpu default to 1).
- `-W/--worksize N` — solver thread count per instance (default 2^20).

## Tested hardware

Rates are Equihash solutions per second (Sol/s), the unit pools use. "Not
recorded" fields are intentionally left open until that machine is retested;
they should not be inferred from the GPU model.

| Test date | Host CPU / memory | GPU | OS, API, backend | Miner settings | Result | Notes |
| --- | --- | --- | --- | --- | ---: | --- |
| 2026-07-30 | Apple M4 Max, 16 cores (12P + 4E), 128 GB | Apple M4 Max, 40 cores | macOS 26.4.1, Metal, wgpu 22.1 | `-I 1 -W 1048576` | 52–55 Sol/s | Local sustained range; controlled 100-nonce solver test: 54.665 Sol/s. AC power with Automatic or High Power mode is required for a representative run. |
| Not recorded | AMD Ryzen AI MAX+ 395, memory not recorded | Radeon 8060S | Linux/driver not recorded, Vulkan/RADV, wgpu | `-I 1` | 47.1 Sol/s | Project measurement. Native HIP tuning uses `-I 1 -W 5592320`; its result still needs recording. |
| 2026-07-30 | Intel Core i9-13900HK, 32 GiB | NVIDIA GeForce RTX 3050 Ti Laptop GPU | Gentoo Linux, NVIDIA 595.71.05, CUDA sm86 | `5620d62`, `-B -I 1`, 30 s, 35 W firmware limit | 18.0 Sol/s | Exact unified artifact; 42/42 solutions verified. The 4 GiB GPU requires one solver instance. |
| 2026-07-30 | AMD EPYC 9654 host, 1 TiB | NVIDIA GeForce RTX 4090 | Gentoo Linux, NVIDIA 595.71.05, CUDA sm89 | `5620d62`, `-B -I 4`, 30 s | 208.9 Sol/s | Exact unified artifact; 42/42 solutions verified. A longer mining sample ranged from 208.8 to 212.2 Sol/s. |
| 2026-07-30 | AMD Ryzen 9 9950X3D, 128 GiB | NVIDIA GeForce RTX 5090 | Fedora 44, NVIDIA 610.43.03, CUDA 13.3 sm120 | `5620d62`, `-B -I 4`, 60 s | 299.4 Sol/s | Exact unified artifact (299.3 Sol/s at 30 s); 42/42 solutions verified. |

For each new result, record the date, CPU and memory, exact GPU/compute-unit
count, OS and driver, backend/API, git commit, power mode, instance/work-size
settings, sample duration, and sustained Sol/s. Start with one instance:

```sh
./gominer-wgpu -l
./gominer-wgpu -B -I 1
```
