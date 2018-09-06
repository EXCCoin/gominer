# gominer
gominer is an application for performing Proof-of-Work (PoW) mining on the
Exchange Coin network. It supports solo and stratum/pool mining using CUDA.

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
You will either need to install CUDA for NVIDIA graphics cards or OpenCL library/headers that support your device such as: AMDGPU-PRO (for newer AMD cards), Beignet (for Intel Graphics), or Catalyst (for older AMD cards).

For example, on Ubuntu 16.04 you can install the necessary OpenCL packages (for Intel Graphics) and CUDA libraries with:
```
sudo apt-get install beignet-dev nvidia-cuda-dev nvidia-cuda-toolkit
```

gominer has been built successfully on Ubuntu 16.04 with go1.6.2, go1.7.1, g++ 5.4.0, and beignet-dev 1.1.1-2 although other combinations should work as well.

#### Instructions
To download and build gominer, run:
```
go get -u github.com/golang/dep/cmd/dep
mkdir -p $GOPATH/src/github.com/decred
cd $GOPATH/src/github.com/decred
git clone  https://github.com/decred/gominer.git
cd gominer
dep ensure
```

For CUDA with NVIDIA Management Library (NVML) support:
```
make
```

## Building on Windows
#### Pre-Requisites
- Download and install the official Go Windows binaries >= v1.9 from [here](https://golang.org/dl/)
  * Make sure you've got properly set `GOROOT` and `GOPATH` environment variables
  * Make sure you've got `%GOPATH%\bin` in your `PATH`
- Download and install Git for Windows from [here](https://git-scm.com/download/win)
  * Make sure to you've got `git` binary (by default: `C:\Program Files\Git\bin`) accessible by `PATH`
- Download and install x64 toolchain of MinGW-w64 from [here](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/installer/mingw-w64-install.exe/download)
  * Select `x86_64` architecture during installation setup
  * Make sure to you've got MinGW-w64 binaries (by default: `C:\Program Files\mingw-w64\x86_64-8.1.0-posix-seh-rt_v6-rev0\mingw64\bin`) accessible by `PATH`
- Install [dep](https://github.com/golang/dep): `go get -u github.com/golang/dep/cmd/dep`
- Download and install Microsoft Visual Studio 2017 Community with v140 toolset
  * Make sure you've got `C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin` in your `PATH`
- Install Nvidia drivers >= v399.07 (you can use Geforce Experience from [here](https://www.nvidia.pl/geforce/geforce-experience/) to do it)
  * Make sure you’ve got `C:\Program Files\NVIDIA Corporation\NVSMI` in your `PATH`
- Install CUDA >= v9.2 from [here](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
  * Make sure you’ve got `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\bin` in your `PATH`

#### Instructions
```
cd $GOPATH/src/github.com/EXCCoin/gominer
build.bat
```
