#include "eqcuda1445/eqcuda1445.h"
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define cudaDeviceProp hipDeviceProp_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaSetDevice hipSetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetErrorString hipGetErrorString
#define GPU_RUNTIME_NAME "HIP"
#else
#include <cuda_runtime.h>
#define GPU_RUNTIME_NAME "CUDA"
#endif
#include <cstdio>
#include <cstring>
#include <cstdint>
static unsigned char g_hdr[180];
#ifdef EQ_PHASE_TIMING
extern "C" void eq_dump_phase_timing(void);
#endif
static int g_ok = 0, g_bad = 0;
static int cb(void *, void *sol) {
    int rc = equihash_verify_c((const char *)g_hdr, sizeof g_hdr, (const unsigned char *)sol);
    rc == 0 ? g_ok++ : g_bad++;
    if (rc != 0) printf("VERIFY FAILED rc=%d\n", rc);
    return 0;
}

static int test_device(int device) {
    cudaDeviceProp prop{};
    cudaError_t err = cudaSetDevice(device);
    if (err == cudaSuccess)
        err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        printf("GPU %d setup failed: %s\n", device, cudaGetErrorString(err));
        return 1;
    }

    EqSolver *s = eq_create(0);
    if (!s) return 1;
    int total = 0, failed = 0;
    g_ok = g_bad = 0;
    for (uint32_t n = 0; n < 20; n++) {
        memset(g_hdr, 0x42, sizeof g_hdr);
        memcpy(&g_hdr[140], &n, 4); // patch nonce like the solver does
        int r = eq_solve(s, g_hdr, sizeof g_hdr, n, cb, nullptr);
        if (r < 0) { printf("GPU %d solve error %d\n", device, r); failed = 1; break; }
        total += r;
    }
    printf("GPU %d (%s), 20 nonces: %d solutions, %d verified OK, %d FAILED\n",
           device, prop.name, total, g_ok, g_bad);
#ifdef EQ_PHASE_TIMING
    eq_dump_phase_timing();
#endif
    eq_destroy(s);
    return failed || g_bad || !g_ok;
}

int main() {
    if (equihash_verify_c(nullptr, 0, nullptr) == 0) {
        printf("Verifier accepted invalid pointers\n");
        return 1;
    }

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        printf("%s device enumeration failed: %s\n", GPU_RUNTIME_NAME, cudaGetErrorString(err));
        return 1;
    }
    if (count < 1) {
        printf("No %s devices found\n", GPU_RUNTIME_NAME);
        return 1;
    }

    int failed = 0;
    for (int device = 0; device < count; device++)
        failed |= test_device(device);
    return failed;
}
