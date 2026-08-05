$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    $target = "x86_64-pc-windows-gnu"
    rustup target add $target
    cargo rustc --manifest-path eqwgpu1445/Cargo.toml --release --locked --target $target --lib --crate-type cdylib

    $rustOutput = "eqwgpu1445/target/$target/release"
    Copy-Item "$rustOutput/eqwgpu1445.dll" . -Force
    Copy-Item "$rustOutput/libeqwgpu1445.dll.a" . -Force

    $hash = (Get-FileHash eqwgpu1445.dll -Algorithm SHA256).Hash.Substring(0, 12).ToLowerInvariant()
    $env:CGO_ENABLED = "1"
    $env:CC = "gcc"
    go build -trimpath -tags wgpu -o gominer-wgpu.exe "-ldflags=-s -w -X main.appBuild=wgpu.$hash"

    foreach ($dll in "libwinpthread-1.dll", "libgcc_s_seh-1.dll", "libstdc++-6.dll") {
        $path = (& gcc "-print-file-name=$dll").Trim()
        if ($path -eq $dll -or !(Test-Path -LiteralPath $path)) {
            throw "MinGW runtime DLL not found: $dll"
        }
        Copy-Item -LiteralPath $path -Destination . -Force
    }
    Write-Host "Built $PSScriptRoot/gominer-wgpu.exe"
} finally {
    Pop-Location
}
