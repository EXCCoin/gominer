// Copyright (c) 2018 The ExchangeCoin team

#pragma once
#include <cstdint>
#include <string>
#include <functional>

#define WN                  (144)      //
#define WK                  (5)        // algorithm parameters, prefixed with W (for Wagner) to reduce include file conflicts
#define BLAKE_PERSONAL      "ZcashPoW"
#define NDIGITS             (WK + 1)
#define DIGITBITS           (WN / NDIGITS)
#define BASE                (1 << DIGITBITS)
#define PROOFSIZE           (1 << WK)
#define NHASHES             (2 * BASE)
#define HASHESPERBLAKE      (512 / WN)
#define HASHOUT             (HASHESPERBLAKE * WN / 8)
#define COMPRESSED_SOL_SIZE (PROOFSIZE * (DIGITBITS + 1) / 8)
#define MAXSOLS             (10)

typedef uint16_t u16;
typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned char uchar;
typedef u32 proof[PROOFSIZE];
typedef uchar cproof[COMPRESSED_SOL_SIZE];

enum class verify_code { POW_OK, POW_HEADER_LENGTH, POW_DUPLICATE, POW_OUT_OF_ORDER, POW_NONZERO_XOR };

inline const char *verify_code_str(verify_code code) {
    switch (code) {
        case verify_code::POW_OK:
            return "OK";
        case verify_code::POW_HEADER_LENGTH:
            return "wrong header length";
        case verify_code::POW_DUPLICATE:
            return "duplicate index";
        case verify_code::POW_OUT_OF_ORDER:
            return "indices out of order";
        case verify_code::POW_NONZERO_XOR:
            return "nonzero xor";
        default:
            return "<unknown_verify_code>";
    }
}


verify_code equihash_verify(const char *header, u64 header_len, u32 nonce, const cproof indices);

verify_code equihash_verify(const std::string &header, u32 nonce, const cproof indices);

extern "C" int equihash_verify_c(const char *header, u64 header_len, u32 nonce, const cproof indices);

verify_code equihash_verify_uncompressed(const char *header, u64 header_len, u32 nonce, const proof indices);

verify_code equihash_verify_uncompressed(const std::string &header, u32 nonce, const proof indices);

extern "C" int equihash_verify_uncompressed_c(const char *header, u64 header_len, u32 nonce, const proof indices);


int equihash_solve(const char *header, u64 header_len,
                   u32 nonce,
                   std::function<void(const cproof)> on_solution_found);

int equihash_solve(const std::string &header,
                   u32 nonce,
                   std::function<void(const cproof)> on_solution_found);

extern "C" int equihash_solve_c(const char *header, u64 header_len,
                                u32 nonce,
                                void (*on_solution_found)(void *user_data, const cproof solution),
                                void *user_data);
