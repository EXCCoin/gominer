// Equihash CUDA solver
// Copyright (c) 2016 John Tromp
// Copyright (c) 2018-2026 The ExchangeCoin team

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaDeviceScheduleYield hipDeviceScheduleYield
#define cudaSetDeviceFlags hipSetDeviceFlags
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyFromSymbol hipMemcpyFromSymbol
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaStream_t hipStream_t
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamDestroy hipStreamDestroy
#define __shfl_sync(mask, value, lane) __shfl(value, lane)
#define __syncwarp() __builtin_amdgcn_wave_barrier()
#if defined(__HIP_DEVICE_COMPILE__) && \
    (!defined(__AMDGCN_WAVEFRONT_SIZE) || __AMDGCN_WAVEFRONT_SIZE != 32)
#error "eqhip1445 requires 32-lane wavefronts"
#endif
#endif

#include "eqcuda1445.cuh"
#include "solver_details.cuh"
#include "big_solver_hip.cuh"
#include "eqcuda1445.h"
#if defined(__HIP_PLATFORM_AMD__)
static constexpr u32 BB_ROUND_TPB = 1024;
#else
static constexpr u32 BB_ROUND_TPB = 512;
#endif

static constexpr u32 bb_round_shared_bytes(u32 round, u32 capacity) {
#if defined(__HIP_PLATFORM_AMD__)
    return 0;
#else
    return (capacity + (round == 4 ? 1 : capacity)) * sizeof(u16);
#endif
}

#ifdef EQ_PHASE_TIMING
extern "C" void eq_dump_phase_timing(void) {
    unsigned long long h[WK + 1][3];
    cudaMemcpyFromSymbol(h, eq_phase_cycles, sizeof(h));
    for (u32 r = 1; r < WK; ++r) {
        if (!h[r][2])
            continue;
        printf("round %u: compact %.0f Mcyc, match %.0f Mcyc, %.1f%% compact (%llu blocks)\n",
               r, h[r][0] / 1e6, h[r][1] / 1e6,
               100.0 * h[r][0] / double(h[r][0] + h[r][1]), h[r][2]);
    }
}
#endif

verify_code equihash_verify_uncompressed(const char *header, u32 header_len, const proof indices) {
    if (duped(indices))
        return verify_code::POW_DUPLICATE;

    blake2b_state ctx;
    setperson(&ctx);
    blake2b_update(&ctx, (uint8_t *)header, header_len);
    uchar hash[WN / 8];
    return verifyrec(&ctx, indices, hash, WK);
}
extern "C" int equihash_verify_uncompressed_c(const char *header, u32 header_len, const proof indices) {
    return static_cast<int>(equihash_verify_uncompressed(header, header_len, indices));
}
extern "C" int equihash_verify_c(const char *header, u32 header_len, const unsigned char *solution) {
    proof sol;
    uncompress_solution(solution, sol);
    return static_cast<int>(equihash_verify_uncompressed(header, header_len, sol));
}

struct EqSolver {
    equi eq;
    equi *device_eq;
    u32 *big0, *big1, *big2, *big3, *big4;
    u32 *counts0, *counts1;
    u64 *candidates;
    u32 *candidate_count;
    u32 tpb;
    cudaStream_t stream;
    equi *host_eq;    // pinned; nsols readback
    proof *host_sols; // pinned; solution readback

    EqSolver(u32 nthreads)
        : eq(nthreads), device_eq(nullptr), big0(nullptr), big1(nullptr), big2(nullptr), big3(nullptr), big4(nullptr),
          counts0(nullptr), counts1(nullptr), candidates(nullptr),
          candidate_count(nullptr),
          tpb(0), stream(nullptr), host_eq(nullptr), host_sols(nullptr) {
        eq.nslots = nullptr;
        eq.sols = nullptr;
    }
};

#define CU_CHECK(call, onfail)                                                        \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "eqcuda1445: %s: %s (%s:%d)\n", #call,                    \
                    cudaGetErrorString(err_), __FILE__, __LINE__);                    \
            onfail;                                                                   \
        }                                                                             \
    } while (0)

#define CU_NEW(call) CU_CHECK(call, { eq_destroy(s); return nullptr; })

extern "C" EqSolver *eq_create(uint32_t nthreads) {
    if (nthreads == 0) {
        // One thread per bucket benchmarked best on Blackwell (stride loops
        // make oversubscription harmless on smaller GPUs); override with
        // --worksize if a card likes something else.
        nthreads = NBUCKETS;
    }
    const u32 tpb = 256;
    nthreads = (nthreads + tpb - 1) / tpb * tpb;

    EqSolver *s = new EqSolver(nthreads);
    s->tpb = tpb;

    CU_NEW(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking));
    CU_NEW(cudaMalloc((void **)&s->big0, size_t(BB_MID_SLOTS) * 5 * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->big1, size_t(BB_SLOTS) * 5 * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->counts0, BB_BUCKETS * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->counts1, BB_MID_BUCKETS * sizeof(u32)));
    // Round-2/3 parents share aligned records with their hashes; round 4
    // keeps only its final word and 64-bit parent.
    CU_NEW(cudaMalloc((void **)&s->big2, size_t(BB_SLOTS) * 3 * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->big3, size_t(BB_SLOTS) * 4 * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->big4, size_t(BB_SLOTS) * 4 * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->candidates, BB_MAX_CANDIDATES * sizeof(u64)));
    CU_NEW(cudaMalloc((void **)&s->candidate_count, sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->eq.sols, MAXSOLS * sizeof(proof)));
    CU_NEW(cudaMalloc((void **)&s->device_eq, sizeof(equi)));
    CU_NEW(cudaMallocHost((void **)&s->host_eq, sizeof(equi)));
    CU_NEW(cudaMallocHost((void **)&s->host_sols, MAXSOLS * sizeof(proof)));
#if !defined(__HIP_PLATFORM_AMD__)
    CU_NEW((cudaFuncSetAttribute(
        bb_round<1, 1, 13, BB_MID_CAPACITY, 12, BB_CAPACITY, 4, 3, 84, 2560, BB_ROUND_TPB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, bb_round_shared_bytes(1, 2560))));
#endif
    return s;
}

extern "C" void eq_destroy(EqSolver *s) {
    if (!s)
        return;
    if (s->stream)
        cudaStreamSynchronize(s->stream);
    cudaFree(s->big0);
    cudaFree(s->big1);
    cudaFree(s->big2);
    cudaFree(s->big3);
    cudaFree(s->big4);
    cudaFree(s->counts0);
    cudaFree(s->counts1);
    cudaFree(s->candidates);
    cudaFree(s->candidate_count);
    cudaFree(s->eq.sols);
    cudaFree(s->device_eq);
    if (s->host_eq)
        cudaFreeHost(s->host_eq);
    if (s->host_sols)
        cudaFreeHost(s->host_sols);
    if (s->stream)
        cudaStreamDestroy(s->stream);
    delete s;
}

extern "C" int eq_solve(EqSolver *s, const void *header, uint32_t header_len, uint32_t nonce,
                        int (*on_solution)(void *user_data, void *solution), void *user_data) {
    if (!s || !header || header_len != 180)
        return -1;

    // Host-side blake2b init over the header (with nonce patched in); resets nsols.
    s->eq.setstate((const uint8_t *)header, header_len, nonce);
#ifdef EQ_DEBUG
    fprintf(stderr, "eq_solve: len=%u nonce=%u buflen=%u counter=%u\n",
            header_len, nonce, (unsigned)s->eq.blake_ctx.buflen, (unsigned)s->eq.blake_ctx.counter);
#endif

    const u32 blocks = s->eq.nthreads / s->tpb;
    CU_CHECK(cudaMemcpyAsync(s->device_eq, &s->eq, sizeof(equi), cudaMemcpyHostToDevice, s->stream), return -2);
    CU_CHECK(cudaMemsetAsync(s->counts1, 0, BB_MID_BUCKETS * sizeof(u32), s->stream), return -2);
    bb_digitH<<<blocks, s->tpb, 0, s->stream>>>(s->device_eq, s->big0, s->counts1);

    CU_CHECK(cudaMemsetAsync(s->counts0, 0, BB_BUCKETS * sizeof(u32), s->stream), return -2);
    bb_round<1, 1, 13, BB_MID_CAPACITY, 12, BB_CAPACITY, 4, 3, 84, 2560, BB_ROUND_TPB>
        <<<BB_MID_BUCKETS * 2, BB_ROUND_TPB, bb_round_shared_bytes(1, 2560), s->stream>>>(
        s->big0, s->big1, s->counts1, s->counts0);

    CU_CHECK(cudaMemsetAsync(s->counts1, 0, BB_BUCKETS * sizeof(u32), s->stream), return -2);
    bb_round<2, 2, 12, BB_CAPACITY, 12, BB_CAPACITY, 3, 2, 60, 2240, BB_ROUND_TPB>
        <<<BB_BUCKETS * 4, BB_ROUND_TPB, bb_round_shared_bytes(2, 2240), s->stream>>>(
        s->big1, s->big4, s->counts0, s->counts1);

    CU_CHECK(cudaMemsetAsync(s->counts0, 0, BB_BUCKETS * sizeof(u32), s->stream), return -2);
    bb_round<3, 2, 12, BB_CAPACITY, 12, BB_CAPACITY, 2, 2, 36, 2240, BB_ROUND_TPB>
        <<<BB_BUCKETS * 4, BB_ROUND_TPB, bb_round_shared_bytes(3, 2240), s->stream>>>(
        s->big4, s->big3, s->counts1, s->counts0);

    CU_CHECK(cudaMemsetAsync(s->counts1, 0, BB_BUCKETS * sizeof(u32), s->stream), return -2);
    bb_round<4, 2, 12, BB_CAPACITY, 12, BB_CAPACITY, 2, 1, 12, 2240, BB_ROUND_TPB>
        <<<BB_BUCKETS * 4, BB_ROUND_TPB, bb_round_shared_bytes(4, 2240), s->stream>>>(
        s->big3, s->big2, s->counts0, s->counts1);

    CU_CHECK(cudaMemsetAsync(s->candidate_count, 0, sizeof(u32), s->stream), return -2);
    bb_final_candidates<<<BB_BUCKETS * 2, 512, 0, s->stream>>>(
        s->big2, s->counts1, s->candidates, s->candidate_count);
    bb_combine_candidates<<<1, 256, 0, s->stream>>>(
        s->device_eq, s->big0,
        s->big1, reinterpret_cast<const u64 *>(s->big4),
        reinterpret_cast<const u64 *>(s->big3), s->big2,
        s->candidates, s->candidate_count);
    CU_CHECK(cudaMemcpyAsync(s->host_eq, s->device_eq, sizeof(equi), cudaMemcpyDeviceToHost, s->stream), return -2);
    CU_CHECK(cudaMemcpyAsync(s->host_sols, s->eq.sols, MAXSOLS * sizeof(proof), cudaMemcpyDeviceToHost, s->stream), return -2);
    CU_CHECK(cudaStreamSynchronize(s->stream), return -2);
    CU_CHECK(cudaGetLastError(), return -2);

    const u32 nsols = s->host_eq->nsols < MAXSOLS ? s->host_eq->nsols : MAXSOLS;
    int found = 0;
    for (u32 i = 0; i < nsols; i++) {
        if (duped(s->host_sols[i]))
            continue;
        cproof csol;
        compress_solution(s->host_sols[i], csol);
        found++;
        if (on_solution && on_solution(user_data, csol))
            break;
    }
    return found;
}
