#!/usr/bin/env bash
# Builds gominer-wgpu: the portable-GPU (Vulkan/Metal/DX12 — NVIDIA, AMD,
# Intel, Apple) variant. Needs only Rust (rustup.rs), no CUDA toolkit.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "${DIR}"

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
command -v cargo >/dev/null || { echo "cargo not found; install via https://rustup.rs" >&2; exit 1; }

RUST_TARGET="${CARGO_BUILD_TARGET:-}"
CARGO_ARGS=(build --release --locked)
RUST_LIB_DIR="eqwgpu1445/target/release"

if [ -n "${RUST_TARGET}" ]; then
    CARGO_ARGS+=(--target "${RUST_TARGET}")
    RUST_LIB_DIR="eqwgpu1445/target/${RUST_TARGET}/release"
fi

case "${RUST_TARGET}" in
    aarch64-unknown-linux-gnu)
        export GOOS=linux
        export GOARCH=arm64
        export CGO_ENABLED=1
        export CC="${CC:-aarch64-linux-gnu-gcc}"
        export CXX="${CXX:-aarch64-linux-gnu-g++}"
        ;;
esac

(cd eqwgpu1445 && cargo "${CARGO_ARGS[@]}")
cp "${RUST_LIB_DIR}/libeqwgpu1445.a" .

# Content hash forces a Go relink when the solver library changes. GNU/Linux
# normally provides sha256sum; macOS provides shasum.
if command -v sha256sum >/dev/null; then
    LIBHASH="$(sha256sum libeqwgpu1445.a | cut -c1-12)"
elif command -v shasum >/dev/null; then
    LIBHASH="$(shasum -a 256 libeqwgpu1445.a | cut -c1-12)"
else
    echo "neither sha256sum nor shasum was found" >&2
    exit 1
fi

OUTPUT="${OUTPUT:-gominer-wgpu}"
go build -trimpath -tags wgpu -o "${OUTPUT}" \
    -ldflags="-s -w -X main.appBuild=wgpu.${LIBHASH}"

echo "Built ${OUTPUT}"
