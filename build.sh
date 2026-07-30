#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "${DIR}"

# Locate the CUDA toolkit: $CUDA_HOME, nvcc on PATH, or common install dirs.
if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc >/dev/null; then
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    else
        for d in "$HOME"/cuda-* /usr/local/cuda /opt/cuda; do
            [ -x "$d/bin/nvcc" ] && CUDA_HOME="$d" && break
        done
    fi
fi
[ -x "${CUDA_HOME:-}/bin/nvcc" ] || { echo "nvcc not found; set CUDA_HOME" >&2; exit 1; }
echo "Using CUDA toolkit: ${CUDA_HOME}"
CXX=${CXX:-g++}
NVCC_HOST_FLAGS=()
if [ -n "${NVCC_CCBIN:-}" ]; then
    NVCC_HOST_FLAGS=(-ccbin "${NVCC_CCBIN}")
fi

# Fat binary: Turing (sm_75) through Blackwell (sm_120), plus PTX for newer
# GPUs. Pascal/Volta users: build with a CUDA 12.x toolkit and override
# GENCODE accordingly.
GENCODE=${GENCODE:-"\
 -gencode arch=compute_75,code=sm_75 \
 -gencode arch=compute_80,code=sm_80 \
 -gencode arch=compute_86,code=sm_86 \
 -gencode arch=compute_89,code=sm_89 \
 -gencode arch=compute_90,code=sm_90 \
 -gencode arch=compute_120,code=sm_120 \
 -gencode arch=compute_120,code=compute_120"}

mkdir -p obj
"${CXX}" -O3 -march=x86-64 -mtune=generic -fPIC -std=c++17 -c eqcuda1445/blake/blake2b.cpp -o obj/blake.o
"${CUDA_HOME}/bin/nvcc" "${NVCC_HOST_FLAGS[@]}" ${GENCODE} -O3 -std=c++17 -allow-unsupported-compiler \
    -Xptxas -O3 -Xcompiler -O3,-fPIC -c eqcuda1445/solver.cu -o obj/solver.o
ar rcs libeqcuda1445.a obj/solver.o obj/blake.o

# ./build.sh test — solve 20 nonces on every GPU and verify every solution
# against the CPU verifier.
if [ "${1:-}" = "test" ]; then
    "${CXX}" -O2 -std=c++17 -I. -I"${CUDA_HOME}/include" eqcuda1445/test_verify.cpp libeqcuda1445.a \
        -o obj/test_verify -L"${CUDA_HOME}/lib64" -lcudart_static -ldl -lrt -lpthread
    ./obj/test_verify
fi

# Stamp the solver library hash into the build: identifies the kernel build
# and, because Go's build cache is content-based, forces a relink whenever
# the .a changes.
LIBHASH="$(sha256sum libeqcuda1445.a | cut -c1-12)"

CGO_CFLAGS="-I${CUDA_HOME}/include ${CGO_CFLAGS:-}" \
CGO_CXXFLAGS="-I${CUDA_HOME}/include ${CGO_CXXFLAGS:-}" \
CGO_LDFLAGS="-L${DIR} -L${CUDA_HOME}/lib64 ${CGO_LDFLAGS:-}" \
    go build -trimpath -ldflags="-s -w -X main.appBuild=cuda.${LIBHASH}"

echo "Built ${DIR}/gominer"
