// Copyright (c) 2018-2026 The ExchangeCoin team

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque per-instance solver context. Each instance owns ~2.7 GB of device
 * memory and its own CUDA stream; create several per GPU to keep it busy. */
typedef struct EqSolver EqSolver;

/* Implemented on the Go side (cgo export); called once per non-duplicate
 * solution with the 100-byte compressed proof. Return nonzero to stop
 * reporting further solutions from this solve. */
int equihashProxyGominer(void *userData, void *solution);

/* Allocates all device buffers for one solver instance on the calling
 * thread's current CUDA device. nthreads == 0 scales to the GPU's SM count.
 * Returns NULL on error (details on stderr). */
EqSolver *eq_create(uint32_t nthreads);
void eq_destroy(EqSolver *solver);

/* Runs one (header, nonce) solve. header must be the >= 144-byte serialized
 * Equihash input (180 bytes for algo v1); nonce is patched into the header's
 * nonce field at byte offset 140. Returns the number of solutions found, or
 * a negative value on CUDA error. */
int eq_solve(EqSolver *solver, const void *header, uint32_t header_len, uint32_t nonce,
             int (*on_solution)(void *user_data, void *solution), void *user_data);

/* Verifies a 100-byte compressed solution against a header; 0 == valid. */
int equihash_verify_c(const char *header, uint32_t header_len, const unsigned char *solution);

#ifdef __cplusplus
}
#endif
