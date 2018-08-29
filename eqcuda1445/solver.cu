// Equihash CUDA solver
// Copyright (c) 2016 John Tromp
// Copyright (c) 2018 The ExchangeCoin team

#include "solver_details.cuh"

verify_code equihash_verify_uncompressed(const char *header, u64 header_len, u32 nonce, const proof indices) {
    if (duped(indices))
        return verify_code::POW_DUPLICATE;

    blake2b_state ctx;
    setheader(&ctx, (const uint8_t *)header, header_len, nonce);
    uchar hash[WN / 8];
    return verifyrec(&ctx, indices, hash, WK);
}

verify_code equihash_verify_uncompressed(const std::string &header, u32 nonce, const proof indices) {
    return equihash_verify_uncompressed(header.c_str(), header.length(), nonce, indices);
}

extern "C" int equihash_verify_uncompressed_c(const char *header, u64 header_len, u32 nonce, const proof indices) {
    return static_cast<int>(equihash_verify_uncompressed(header, header_len, nonce, indices));
}

verify_code equihash_verify(const char *header, u64 header_len, u32 nonce, const cproof indices) {
    proof sol;
    uncompress_solution(indices, sol);
    return equihash_verify_uncompressed(header, header_len, nonce, sol);
}

verify_code equihash_verify(const std::string &header, u32 nonce, const cproof indices) {
    return equihash_verify(header.c_str(), header.length(), nonce, indices);
}

extern "C" int equihash_verify_c(const char *header, u64 header_len, u32 nonce, const cproof indices) {
    return static_cast<int>(equihash_verify(header, header_len, nonce, indices));
}

int equihash_solve(const char *header, u64 header_len, u32 nonce, std::function<void(const cproof)> on_solution_found) {
#define printf                                                                                                         \
    if (debug_logs)                                                                                                    \
    printf

    bool debug_logs = false;
    u64 nthreads = 8192;
    u64 tpb = 128; // threads per block, roughly square root of nthreads
    u64 range = 1;

    if (debug_logs) {
        std::string header_hex = to_hex((const unsigned char *)header, header_len);
        printf("Looking for wagner-tree on (\"%s\",%ui", header_hex.c_str(), nonce);
    }

    if (range > 1)
        printf("-%lu", nonce + range - 1);

    printf(") with %d %d-bits digits and %li threads (%li per block)\n", NDIGITS, DIGITBITS, nthreads, tpb);
    equi eq(static_cast<u32>(nthreads));

    u32 *heap0, *heap1;
    checkCudaErrors(cudaMalloc((void **)&heap0, sizeof(digit0)));
    checkCudaErrors(cudaMalloc((void **)&heap1, sizeof(digit1)));
    for (u32 r = 0; r < WK; r++)
        if ((r & 1) == 0)
            eq.hta.trees0[r / 2] = (bucket0 *)(heap0 + r / 2);
        else
            eq.hta.trees1[r / 2] = (bucket1 *)(heap1 + r / 2);

    checkCudaErrors(cudaMalloc((void **)&eq.nslots, 2 * NBUCKETS * sizeof(u32)));
    checkCudaErrors(cudaMemset((void *)eq.nslots, 0, 2 * NBUCKETS * sizeof(u32)));
    checkCudaErrors(cudaMalloc((void **)&eq.sols, MAXSOLS * sizeof(proof)));

    equi *device_eq;
    checkCudaErrors(cudaMalloc((void **)&device_eq, sizeof(equi)));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    proof sols[MAXSOLS];
    u32 sumnsols = 0;
    for (int r = 0; r < range; r++) {
        cudaEventRecord(start, NULL);
        eq.setheadernonce((const uint8_t *)header, header_len, nonce);

        printf("eq.blake_ctx.buf: ");
        for (int i = 0; i < sizeof(eq.blake_ctx.buf); i++)
            printf("%c(%d) ", char(eq.blake_ctx.buf[i]), int(eq.blake_ctx.buf[i]));
        printf("\n");

        checkCudaErrors(cudaMemcpy(device_eq, &eq, sizeof(equi), cudaMemcpyHostToDevice));
        digitH<<<nthreads / tpb, tpb>>>(device_eq);
        eq.showbsizes(0);
#if BUCKBITS == 16 && RESTBITS == 4 && defined XINTREE && defined(UNROLL)
        digit_1<<<nthreads / tpb, tpb>>>(device_eq);
        eq.showbsizes(1);
        digit2<<<nthreads / tpb, tpb>>>(device_eq);
        eq.showbsizes(2);
        digit3<<<nthreads / tpb, tpb>>>(device_eq);
        eq.showbsizes(3);
        digit4<<<nthreads / tpb, tpb>>>(device_eq);
        eq.showbsizes(4);
        digit5<<<nthreads / tpb, tpb>>>(device_eq);
        eq.showbsizes(5);
        digit6<<<nthreads / tpb, tpb>>>(device_eq);
        eq.showbsizes(6);
        digit7<<<nthreads / tpb, tpb>>>(device_eq);
        eq.showbsizes(7);
        digit8<<<nthreads / tpb, tpb>>>(device_eq);
        eq.showbsizes(8);
#else
        for (u32 r = 1; r < WK; r++) {
            r & 1 ? digitO<<<nthreads / tpb, tpb>>>(device_eq, r) : digitE<<<nthreads / tpb, tpb>>>(device_eq, r);
            eq.showbsizes(r);
        }
#endif
        digitK<<<nthreads / tpb, tpb>>>(device_eq);

        checkCudaErrors(cudaMemcpy(&eq, device_eq, sizeof(equi), cudaMemcpyDeviceToHost));
        u32 maxsols = min(MAXSOLS, eq.nsols);
        checkCudaErrors(cudaMemcpy(sols, eq.sols, maxsols * sizeof(proof), cudaMemcpyDeviceToHost));
        cudaEventRecord(stop, NULL);
        cudaEventSynchronize(stop);
        float duration;
        cudaEventElapsedTime(&duration, start, stop);
        printf("%d rounds completed in %.3f seconds.\n", WK, duration / 1000.0f);

        u32 s, nsols, ndupes;
        for (s = nsols = ndupes = 0; s < maxsols; s++) {
            if (duped(sols[s])) {
                ndupes++;
                continue;
            }
            nsols++;
            if (on_solution_found) {
                cproof csol;
                compress_solution(sols[s], csol);
                on_solution_found(csol);
            }
        }
        printf("%d solutions %d dupes\n", nsols, ndupes);
        sumnsols += nsols;
    }
    checkCudaErrors(cudaFree(eq.nslots));
    checkCudaErrors(cudaFree(eq.sols));
    checkCudaErrors(cudaFree(eq.hta.trees0[0]));
    checkCudaErrors(cudaFree(eq.hta.trees1[0]));

    printf("%d total solutions\n", sumnsols);

#undef printf
    return 0;
}

int equihash_solve(const std::string &header, u32 nonce, std::function<void(const cproof)> on_solution_found) {
    return equihash_solve(header.c_str(), header.length(), nonce, on_solution_found);
}

extern "C" int equihash_solve_c(const char *header, u64 header_len, u32 nonce,
                                void (*on_solution_found)(void *user_data, const cproof solution), void *user_data) {
    return equihash_solve(header, header_len, nonce,
                          [=](const cproof solution) { on_solution_found(user_data, solution); });
}
