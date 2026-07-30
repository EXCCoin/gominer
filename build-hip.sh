#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "${DIR}"

HIPCC="${HIPCC:-$(command -v hipcc || true)}"
[ -x "${HIPCC}" ] || { echo "hipcc not found; install ROCm HIP or set HIPCC" >&2; exit 1; }
HIP_PATH="${HIP_PATH:-$(hipconfig --path)}"
GPU_ARCH="${GPU_ARCH:-native}"
echo "Using HIP from ${HIP_PATH} for ${GPU_ARCH}"

mkdir -p obj/hip
g++ -O3 -march=x86-64 -mtune=generic -fPIC -std=c++17 \
    -c eqcuda1445/blake/blake2b.cpp -o obj/hip/blake.o
"${HIPCC}" -O2 -w -std=c++17 --offload-arch="${GPU_ARCH}" -fPIC \
    -D__HIP_PLATFORM_AMD__ \
    '-D__forceinline__=inline __attribute__((always_inline))' \
    -Ieqcuda1445 -c eqcuda1445/solver_hip.cu -o obj/hip/solver.o
ar rcs libeqhip1445.a obj/hip/solver.o obj/hip/blake.o

if [ "${1:-}" = "test" ]; then
    "${HIPCC}" -O2 -std=c++17 -I. --offload-arch="${GPU_ARCH}" \
        -D__HIP_PLATFORM_AMD__ -c eqcuda1445/test_verify.cpp \
        -o obj/hip/test_verify.o
    "${HIPCC}" --offload-arch="${GPU_ARCH}" obj/hip/test_verify.o \
        libeqhip1445.a -o obj/hip/test_verify
    ./obj/hip/test_verify
fi

LIBHASH="$(sha256sum libeqhip1445.a | cut -c1-12)"
OUTPUT="${OUTPUT:-gominer-hip}"
CGO_CFLAGS="-D__HIP_PLATFORM_AMD__ -I${HIP_PATH}/include ${CGO_CFLAGS:-}" \
CGO_CXXFLAGS="-I${HIP_PATH}/include ${CGO_CXXFLAGS:-}" \
CGO_LDFLAGS="-L${DIR} -L${HIP_PATH}/lib -L${HIP_PATH}/lib64 ${CGO_LDFLAGS:-}" \
GOCACHE="${GOCACHE:-${TMPDIR:-/tmp}/gominer-gocache}" \
GOFLAGS="${GOFLAGS:--mod=readonly}" \
GOTOOLCHAIN="${GOTOOLCHAIN:-auto}" \
    go build -trimpath -tags hip -o "${OUTPUT}" \
    -ldflags="-s -w -X main.appBuild=hip.${LIBHASH}"

echo "Built ${DIR}/${OUTPUT}"
