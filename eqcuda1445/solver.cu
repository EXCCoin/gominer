// Equihash CUDA solver
// Copyright (c) 2016 John Tromp
// Copyright (c) 2018-2026 The ExchangeCoin team

#include "eqcuda1445.cuh"
#include "solver_details.cuh"

namespace modern_cuda {
#include "big_solver.cuh"
}

#undef EQ_BB_LEAF_BUCKET_BITS
#undef EQ_BB_LEAF_CAPACITY
#undef EQ_BB_LEAF_SLOT_BITS
#undef EQ_BB_MID_BUCKET_BITS
#undef EQ_BB_MID_CAPACITY
#undef EQ_BB_MID_SLOT_BITS
#undef EQ_BB_SHARED_PRECOMPUTE
#define EQ_BB_LEAF_BUCKET_BITS 12
#define EQ_BB_LEAF_CAPACITY 8688
#define EQ_BB_LEAF_SLOT_BITS 14
#define EQ_BB_MID_BUCKET_BITS 13
#define EQ_BB_MID_CAPACITY 4592
#define EQ_BB_MID_SLOT_BITS 13
#define EQ_BB_SHARED_PRECOMPUTE 0
namespace ampere_cuda {
#include "big_solver.cuh"
}

#undef EQ_BB_LEAF_BUCKET_BITS
#undef EQ_BB_LEAF_CAPACITY
#undef EQ_BB_LEAF_SLOT_BITS
#undef EQ_BB_MID_BUCKET_BITS
#undef EQ_BB_MID_CAPACITY
#undef EQ_BB_MID_SLOT_BITS
#undef EQ_BB_SHARED_PRECOMPUTE
#include "eqcuda1445.h"

#ifdef EQ_PHASE_TIMING
extern "C" void eq_dump_phase_timing(void) {
    unsigned long long h[WK + 1][3];
    cudaMemcpyFromSymbol(h, modern_cuda::eq_phase_cycles, sizeof(h));
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
    u32 *big0, *big1, *big2;
    u32 *counts0, *counts1;
    u64 *parents[WK - 1];
    u32 tpb;
    bool ampere;
    cudaStream_t stream;
    equi *host_eq;    // pinned; nsols readback
    proof *host_sols; // pinned; solution readback

    EqSolver(u32 nthreads)
        : eq(nthreads), device_eq(nullptr), big0(nullptr), big1(nullptr), big2(nullptr),
          counts0(nullptr), counts1(nullptr), parents{},
          tpb(0), ampere(false), stream(nullptr), host_eq(nullptr), host_sols(nullptr) {
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

static constexpr bool use_ampere_solver(int major, int minor) {
    return major == 8 && minor == 6;
}

static_assert(use_ampere_solver(8, 6));
static_assert(!use_ampere_solver(8, 9));
static_assert(!use_ampere_solver(12, 0));

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

    int device = 0;
    cudaDeviceProp properties{};
    CU_NEW(cudaGetDevice(&device));
    CU_NEW(cudaGetDeviceProperties(&properties, device));
    s->ampere = use_ampere_solver(properties.major, properties.minor);

    const size_t leaf_slots = s->ampere ? ampere_cuda::BB_LEAF_SLOTS
                                        : modern_cuda::BB_LEAF_SLOTS;
    const size_t mid_slots = s->ampere ? ampere_cuda::BB_MID_SLOTS
                                       : modern_cuda::BB_MID_SLOTS;
    const u32 leaf_buckets = s->ampere ? ampere_cuda::BB_LEAF_BUCKETS
                                       : modern_cuda::BB_LEAF_BUCKETS;
    const u32 mid_buckets = s->ampere ? ampere_cuda::BB_MID_BUCKETS
                                      : modern_cuda::BB_MID_BUCKETS;
    CU_NEW(cudaMalloc((void **)&s->big0, leaf_slots * 5 * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->big1, mid_slots * 4 * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->counts0, leaf_buckets * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->counts1, mid_buckets * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->parents[0], mid_slots * sizeof(u64)));
    // Round-2/3 parents share aligned records with their hashes; round 4
    // writes uint4 records (final word + parent) into big2.
    CU_NEW(cudaMalloc((void **)&s->big2, leaf_slots * 4 * sizeof(u32)));
    CU_NEW(cudaMalloc((void **)&s->eq.sols, MAXSOLS * sizeof(proof)));
    CU_NEW(cudaMalloc((void **)&s->device_eq, sizeof(equi)));
    CU_NEW(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking));
    CU_NEW(cudaMallocHost((void **)&s->host_eq, sizeof(equi)));
    CU_NEW(cudaMallocHost((void **)&s->host_sols, MAXSOLS * sizeof(proof)));
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
    cudaFree(s->counts0);
    cudaFree(s->counts1);
    for (u32 r = 0; r < WK - 1; ++r)
        cudaFree(s->parents[r]);
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
    if (s->ampere) {
        using namespace ampere_cuda;
        CU_CHECK(cudaMemsetAsync(s->counts0, 0, BB_LEAF_BUCKETS * sizeof(u32), s->stream), return -2);
        bb_digitH<<<blocks, s->tpb, 0, s->stream>>>(s->device_eq, s->big0, s->counts0);

        CU_CHECK(cudaMemsetAsync(s->counts1, 0, BB_MID_BUCKETS * sizeof(u32), s->stream), return -2);
        bb_round<1, 2, 12, BB_CAPACITY, 13, BB_MID_CAPACITY, 4, 3, 83, 2240, 512>
            <<<BB_BUCKETS * 4, 512, 0, s->stream>>>(
            s->big0, s->big1, s->counts0, s->counts1, s->parents[0]);

        CU_CHECK(cudaMemsetAsync(s->counts0, 0, BB_BUCKETS * sizeof(u32), s->stream), return -2);
        bb_round<2, 1, 13, BB_MID_CAPACITY, 12, BB_CAPACITY, 3, 2, 60, 2808, 512>
            <<<BB_MID_BUCKETS * 2, 512, 0, s->stream>>>(
            s->big1, s->big0, s->counts1, s->counts0, nullptr);

        CU_CHECK(cudaMemsetAsync(s->counts1, 0, BB_MID_BUCKETS * sizeof(u32), s->stream), return -2);
        bb_round<3, 2, 12, BB_CAPACITY, 13, BB_MID_CAPACITY, 2, 2, 35, 2240, 512>
            <<<BB_BUCKETS * 4, 512, 0, s->stream>>>(
            s->big0, s->big1, s->counts0, s->counts1, nullptr);

        CU_CHECK(cudaMemsetAsync(s->counts0, 0, BB_BUCKETS * sizeof(u32), s->stream), return -2);
        bb_round<4, 1, 13, BB_MID_CAPACITY, 12, BB_CAPACITY, 2, 1, 12, 2816, 512>
            <<<BB_MID_BUCKETS * 2, 512, 0, s->stream>>>(
            s->big1, s->big2, s->counts1, s->counts0, nullptr);

        bb_final<0><<<BB_BUCKETS, s->tpb, 0, s->stream>>>(
            s->device_eq, s->big2, s->counts0, s->big0,
            s->parents[0], reinterpret_cast<const u64 *>(s->big0),
            reinterpret_cast<const u64 *>(s->big1));
    } else {
        using namespace modern_cuda;
        CU_CHECK(cudaMemsetAsync(s->counts0, 0, BB_LEAF_BUCKETS * sizeof(u32), s->stream), return -2);
        bb_digitH<<<blocks, s->tpb, 0, s->stream>>>(s->device_eq, s->big0, s->counts0);

        CU_CHECK(cudaMemsetAsync(s->counts1, 0, BB_MID_BUCKETS * sizeof(u32), s->stream), return -2);
        bb_round<1, 0, BB_LEAF_BUCKET_BITS, BB_LEAF_CAPACITY,
                 BB_MID_BUCKET_BITS, BB_MID_CAPACITY, 4, 3, 82, BB_LEAF_CAPACITY, 512>
            <<<BB_LEAF_BUCKETS, 512, 0, s->stream>>>(
            s->big0, s->big1, s->counts0, s->counts1, s->parents[0]);

        CU_CHECK(cudaMemsetAsync(s->counts0, 0, BB_BUCKETS * sizeof(u32), s->stream), return -2);
        bb_round<2, 0, BB_MID_BUCKET_BITS, BB_MID_CAPACITY,
                 BB_BUCKET_BITS, BB_CAPACITY, 3, 2, 58, BB_MID_CAPACITY, 512>
            <<<BB_MID_BUCKETS, 512, 0, s->stream>>>(
            s->big1, s->big0, s->counts1, s->counts0, nullptr);

        CU_CHECK(cudaMemsetAsync(s->counts1, 0, BB_MID_BUCKETS * sizeof(u32), s->stream), return -2);
        bb_round<3, 0, BB_BUCKET_BITS, BB_CAPACITY,
                 BB_MID_BUCKET_BITS, BB_MID_CAPACITY, 2, 2, 34, BB_CAPACITY, 512>
            <<<BB_BUCKETS, 512, 0, s->stream>>>(
            s->big0, s->big1, s->counts0, s->counts1, nullptr);

        CU_CHECK(cudaMemsetAsync(s->counts0, 0, BB_BUCKETS * sizeof(u32), s->stream), return -2);
        bb_round<4, 0, BB_MID_BUCKET_BITS, BB_MID_CAPACITY,
                 BB_BUCKET_BITS, BB_CAPACITY, 2, 1, 10, BB_CAPACITY, 512>
            <<<BB_MID_BUCKETS, 512, 0, s->stream>>>(
            s->big1, s->big2, s->counts1, s->counts0, nullptr);

        bb_final<0><<<BB_BUCKETS, s->tpb, 0, s->stream>>>(
            s->device_eq, s->big2, s->counts0, s->big0,
            s->parents[0], reinterpret_cast<const u64 *>(s->big0),
            reinterpret_cast<const u64 *>(s->big1));
    }

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
