
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
- Download and install Go >= v1.10 from [here](https://golang.org/dl/)
  * Make sure you've got properly set `GOROOT` and `GOPATH` environment variables
  * Make sure you've got `$GOPATH\bin` in your `PATH`
- Install [dep](https://github.com/golang/dep): `go get -u github.com/golang/dep/cmd/dep`
- Install Nvidia drivers >= v396.37
- Install CUDA >= v9.2 from [here](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64) (you can follow [this](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) instruction)
  * Add those lines to your `.bashrc` file:
	```
	export CUDA_HOME=/usr/local/cuda-9.2
	export PATH=${CUDA_HOME}/bin${PATH:+:${PATH}}
	export C_INCLUDE_PATH=${CUDA_HOME}/include:${C_INCLUDE_PATH}
	export LD_LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
	export LIBRARY_PATH=${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
	export CUDACXX=$(which nvcc)
	```
- Reload your `ld` cache: `sudo rm /etc/ld.so.cache && sudo ldconfig && sudo ldconfig -v`

#### Instructions
```
go get github.com/EXCCoin/gominer
cd $GOPATH/src/github.com/EXCCoin/gominer
./build.sh
```

## Building on Windows
#### Pre-Requisites
- Download and install Go >= v1.10 from [here](https://golang.org/dl/)
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
go get github.com/EXCCoin/gominer
cd $GOPATH/src/github.com/EXCCoin/gominer
build.bat
```
