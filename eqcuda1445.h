#ifndef EQCUDA1445_H
#define EQCUDA1445_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int equihashProxy(void *blockData, void *solution);

int equihash_solve_c(const char *header, uint64_t header_len,
                     uint32_t nonce,
                     void (*on_solution_found)(void *user_data, const unsigned char solution[100]),
                     void *user_data);

inline int EquihashSolveCuda(const void *header, uint64_t header_len, uint32_t nonce, void *user_data) {
    return equihash_solve_c((const char *)header, header_len, nonce, (void (*)(void *, const unsigned char[100]))(equihashProxy), user_data);
}

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* EQCUDA1445_H */