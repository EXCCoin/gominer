#!/usr/bin/env bash
# Builds gominer-wgpu: the portable-GPU (Vulkan/Metal/DX12 — NVIDIA, AMD,
# Intel, Apple) variant. Needs only Rust (rustup.rs), no CUDA toolkit.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
cd "${DIR}"

[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"
command -v cargo >/dev/null || { echo "cargo not found; install via https://rustup.rs" >&2; exit 1; }

(cd eqwgpu1445 && cargo build --release)
cp eqwgpu1445/target/release/libeqwgpu1445.a .

# Content hash forces a Go relink when the solver library changes.
LIBHASH="$(sha256sum libeqwgpu1445.a | cut -c1-12)"

go build -trimpath -tags wgpu -o gominer-wgpu \
    -ldflags="-s -w -X main.appBuild=wgpu.${LIBHASH}"

echo "Built ${DIR}/gominer-wgpu"
