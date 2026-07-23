// Equihash CUDA solver
// Copyright (c) 2016 John Tromp
// Copyright (c) 2018-2026 The ExchangeCoin team

#include "eqcuda1445.cuh"
#include "solver_details.cuh"
#include "eqcuda1445.h"

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
    u32 *heap0, *heap1;
    u32 tpb;
    cudaStream_t stream;
    equi *host_eq;    // pinned; nsols readback
    proof *host_sols; // pinned; solution readback

    EqSolver(u32 nthreads)
        : eq(nthreads), device_eq(nullptr), heap0(nullptr), heap1(nullptr),
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

    CU_NEW(cudaMalloc((void **)&s->heap0, sizeof(digit0)));
    CU_NEW(cudaMalloc((void **)&s->heap1, sizeof(digit1)));
    for (u32 r = 0; r < WK; r++)
        if ((r & 1) == 0)
            s->eq.hta.trees0[r / 2] = (bucket0 *)(s->heap0 + r / 2);
        else
            s->eq.hta.trees1[r / 2] = (bucket1 *)(s->heap1 + r / 2);

    CU_NEW(cudaMalloc((void **)&s->eq.nslots, 2 * NBUCKETS * sizeof(u32)));
    CU_NEW(cudaMemset((void *)s->eq.nslots, 0, 2 * NBUCKETS * sizeof(u32)));
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
    cudaFree(s->heap0);
    cudaFree(s->heap1);
    cudaFree((void *)s->eq.nslots);
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
    if (!s || !header || header_len < 144 || header_len > 512)
        return -1;

    // Host-side blake2b init over the header (with nonce patched in); resets nsols.
    s->eq.setstate((const uint8_t *)header, header_len, nonce);
#ifdef EQ_DEBUG
    fprintf(stderr, "eq_solve: len=%u nonce=%u buflen=%u counter=%u\n",
            header_len, nonce, (unsigned)s->eq.blake_ctx.buflen, (unsigned)s->eq.blake_ctx.counter);
#endif

    const u32 blocks = s->eq.nthreads / s->tpb;
    CU_CHECK(cudaMemcpyAsync(s->device_eq, &s->eq, sizeof(equi), cudaMemcpyHostToDevice, s->stream), return -2);
    digitH<<<blocks, s->tpb, 0, s->stream>>>(s->device_eq);
    digitRT<1><<<blocks, s->tpb, 0, s->stream>>>(s->device_eq);
    digitRT<2><<<blocks, s->tpb, 0, s->stream>>>(s->device_eq);
    digitRT<3><<<blocks, s->tpb, 0, s->stream>>>(s->device_eq);
    digitRT<4><<<blocks, s->tpb, 0, s->stream>>>(s->device_eq);
    digitK<<<blocks, s->tpb, 0, s->stream>>>(s->device_eq);

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
