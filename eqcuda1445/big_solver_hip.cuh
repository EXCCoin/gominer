// Large-bucket Equihash (144,5) CUDA dataflow.
#pragma once

#ifdef EQ_PHASE_TIMING
// [round][0]=init+compact cycles, [1]=match+scatter cycles, [2]=blocks
__device__ unsigned long long eq_phase_cycles[WK + 1][3];
#endif

static constexpr u32 BB_BUCKET_BITS = 12;
static constexpr u32 BB_BUCKETS = 1u << BB_BUCKET_BITS;
static constexpr u32 BB_CAPACITY = 8688;
static constexpr u32 BB_SLOTS = BB_BUCKETS * BB_CAPACITY;
static constexpr u32 BB_MID_BUCKET_BITS = 13;
static constexpr u32 BB_MID_BUCKETS = 1u << BB_MID_BUCKET_BITS;
static constexpr u32 BB_MID_CAPACITY = 4592;
static constexpr u32 BB_MID_SLOTS = BB_MID_BUCKETS * BB_MID_CAPACITY;
static constexpr u32 BB_LEAF_MASK = (1u << (DIGITBITS + 1)) - 1;

__device__ __forceinline__ u32 bb_leaf_get(const u32 *data, u32 index, u32 word) {
    return data[index * 5 + word];
}
__device__ __forceinline__ u32 bb_be32(const uchar *p) {
#if defined(__HIP_PLATFORM_AMD__)
    u32 value;
    memcpy(&value, p, sizeof(value));
    return __builtin_bswap32(value);
#else
    return (u32(p[0]) << 24) | (u32(p[1]) << 16) | (u32(p[2]) << 8) | p[3];
#endif
}

// Bucket by the first 13 bits. The five stored words contain the remaining
// 131 hash bits, left-aligned; the unused low bits carry the leaf index.
__global__ __launch_bounds__(256) void bb_digitH(equi *eq, u32 *out, u32 *counts) {
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ blake2b_pre pre;
    if (threadIdx.x == 0)
        blake2b_precompute(&eq->blake_ctx, &pre);
    __syncthreads();
    uchar hash[HASHOUT];

    for (u32 block = id; block < NBLOCKS; block += eq->nthreads) {
        blake2b_gpu_hash_pre(&pre, block, hash);
        u32 firsts[HASHESPERBLAKE];
        u32 slots[HASHESPERBLAKE];
#pragma unroll
        for (u32 i = 0; i < HASHESPERBLAKE; ++i) {
            const u32 leaf = block * HASHESPERBLAKE + i;
            if (leaf >= NHASHES) {
                slots[i] = BB_MID_CAPACITY;
                continue;
            }
            const uchar *h = hash + i * WN / 8;
            const u32 a0 = bb_be32(h);
            firsts[i] = a0;
            slots[i] = atomicAdd(&counts[a0 >> 19], 1);
        }
#pragma unroll
        for (u32 i = 0; i < HASHESPERBLAKE; ++i) {
            if (slots[i] >= BB_MID_CAPACITY)
                continue;

            const u32 leaf = block * HASHESPERBLAKE + i;
            const uchar *h = hash + i * WN / 8;
            const u32 a0 = firsts[i];
            const u32 a1 = bb_be32(h + 4);
            const u32 a2 = bb_be32(h + 8);
            const u32 a3 = bb_be32(h + 12);
            const u32 a4 = (u32(h[16]) << 24) | (u32(h[17]) << 16);
            const u32 index = (a0 >> 19) * BB_MID_CAPACITY + slots[i];
            u32 *record = out + index * 5;
            record[0] = (a0 << 13) | (a1 >> 19);
            record[1] = (a1 << 13) | (a2 >> 19);
            record[2] = (a2 << 13) | (a3 >> 19);
            record[3] = (a3 << 13) | (a4 >> 19);
            record[4] = (a4 << 13) | leaf;
        }
    }
}

template <u32 ROUND, u32 PART_BITS,
          u32 IN_BUCKET_BITS, u32 IN_CAPACITY,
          u32 OUT_BUCKET_BITS, u32 OUT_CAPACITY,
          u32 IN_WORDS, u32 OUT_WORDS, u32 OUT_BITS, u32 SELECT_CAPACITY,
          u32 TPB = 256>
__global__ __launch_bounds__(TPB) void bb_round(const u32 *__restrict__ in,
                                                 u32 *__restrict__ out,
                                                 const u32 *__restrict__ in_counts,
                                                 u32 *__restrict__ out_counts) {
    static_assert(ROUND >= 1 && ROUND < WK, "collision round");
    constexpr u32 COLLISION_BITS = DIGITBITS - IN_BUCKET_BITS;
    constexpr u32 SHIFT = COLLISION_BITS + OUT_BUCKET_BITS;
    static_assert(PART_BITS <= COLLISION_BITS, "partition width");
    static_assert(SHIFT < 32, "single-word shift");
    constexpr u32 PARTS = 1u << PART_BITS;
    constexpr u32 KEY_BITS = COLLISION_BITS - PART_BITS;
    constexpr u32 KEYS = 1u << KEY_BITS;
    constexpr u32 KEY_MASK = KEYS - 1;
    constexpr u32 LAST_BITS = OUT_BITS & 31;
    constexpr u32 LAST_MASK = LAST_BITS == 0 ? ~0u : (~0u << (32 - LAST_BITS));
    __shared__ u32 heads[KEYS];
    __shared__ u32 selected;
    __shared__ u32 hashes[SELECT_CAPACITY * IN_WORDS];
#if defined(__HIP_PLATFORM_AMD__)
    __shared__ u16 slots[ROUND == 4 ? 1 : SELECT_CAPACITY];
    __shared__ u16 next[SELECT_CAPACITY];
#else
    extern __shared__ u16 links[];
    u16 *slots = links;
    u16 *next = slots + (ROUND == 4 ? 1 : SELECT_CAPACITY);
#endif
    __shared__ uchar drop_hi[ROUND == 3 ? SELECT_CAPACITY : 1];

    const u32 part = blockIdx.x & (PARTS - 1);
    const u32 bucket = blockIdx.x >> PART_BITS;
    const u32 tid = threadIdx.x;
    const u32 n = min(in_counts[bucket], IN_CAPACITY);
    const u32 base = bucket * IN_CAPACITY;
    auto load = [&](u32 s) {
        constexpr u32 IN_STRIDE = ROUND == 1 || ROUND == 2 ? 5 : 4;
        const u32 *record = in + (base + s) * IN_STRIDE;
        if constexpr (IN_STRIDE == 4)
            return *reinterpret_cast<const uint4 *>(record);
        return make_uint4(record[0], record[1], record[2], record[3]);
    };

    for (u32 k = tid; k < KEYS; k += blockDim.x)
        heads[k] = ~0u;
    if (tid == 0)
        selected = 0;
    __syncthreads();

#ifdef EQ_PHASE_TIMING
    const u32 eq_t0 = u32(clock64());
#endif

    // Compact this partition into shared memory once; collision pairs then
    // reuse the staged hashes instead of issuing random global rereads.
    // A vector load serves both the partition-key check and the staged data.
    auto stage = [&](const uint4 rec, const u32 s) {
        const u32 key = rec.x >> (32 - COLLISION_BITS);
#if defined(__HIP_PLATFORM_AMD__)
        const bool matched = (key >> KEY_BITS) == part;
        const u32 votes = u32(__ballot(matched));
        if (!votes)
            return;
        const u32 lane = __lane_id();
        const u32 leader = __ffs(votes) - 1;
        u32 pos = __mbcnt_lo(u32(votes), 0);
        u32 base_pos = 0;
        if (lane == leader)
            base_pos = atomicAdd(&selected, __popc(votes));
        pos += __shfl(base_pos, leader);
        if (!matched)
            return;
#else
        if ((key >> KEY_BITS) != part)
            return;
        const u32 pos = atomicAdd(&selected, 1);
#endif
        if (pos >= SELECT_CAPACITY)
            return;
        if constexpr (ROUND < 4)
            slots[pos] = s;
        if constexpr (ROUND == 1) {
            // Word 0's key bits cancel within a chain, so its top bits are
            // free to carry the 3 tail bits — their pair XOR then falls out
            // of x[0] with no tails[] array.
            hashes[pos * IN_WORDS] = (rec.x & 0x001fffffu) |
                                     (bb_leaf_get(in, base + s, 4) & 0xe0000000u);
            hashes[pos * IN_WORDS + 1] = rec.y;
            hashes[pos * IN_WORDS + 2] = rec.z;
            hashes[pos * IN_WORDS + 3] = rec.w;
        } else if constexpr (ROUND == 2) {
            const u32 dropped = ((rec.z & 0xfffu) << 12) | (rec.w >> 20);
            hashes[pos * IN_WORDS] = (rec.x & 0x000fffffu) | ((dropped >> 12) << 20);
            hashes[pos * IN_WORDS + 1] = rec.y;
            hashes[pos * IN_WORDS + 2] = (rec.z & 0xfffff000u) | (dropped & 0xfff);
        } else if constexpr (ROUND == 3) {
            const u32 dropped = u32(((u64(rec.w) << 32) | rec.z) >> 40);
            hashes[pos * IN_WORDS] = (rec.x & 0x000fffffu) | ((dropped & 0xfff) << 20);
            hashes[pos * IN_WORDS + 1] = rec.y | ((dropped >> 12) & 0xf);
            drop_hi[pos] = dropped >> 16;
        } else if constexpr (ROUND == 4) {
            const u64 meta = u64(s) | (u64(rec.y & 0xffffff) << 14);
            hashes[pos * IN_WORDS] = (rec.x & 0x003fffffu) | (u32(meta >> 28) << 22);
            hashes[pos * IN_WORDS + 1] = (rec.y & 0xf0000000u) | (u32(meta) & 0x0fffffffu);
        }
        next[pos] = (u16)atomicExch(&heads[key & KEY_MASK], pos);
    };
#if defined(__HIP_PLATFORM_AMD__)
    for (u32 s = tid; s < n; s += 9 * blockDim.x) {
        const uint4 recA = load(s);
        const u32 sB = s + blockDim.x;
        const u32 sC = sB + blockDim.x;
        const u32 sD = sC + blockDim.x;
        const u32 sE = sD + blockDim.x;
        const u32 sF = sE + blockDim.x;
        const u32 sG = sF + blockDim.x;
        const u32 sH = sG + blockDim.x;
        const u32 sI = sH + blockDim.x;
        uint4 recB, recC, recD, recE, recF, recG, recH, recI;
        if (sB < n) recB = load(sB);
        if (sC < n) recC = load(sC);
        if (sD < n) recD = load(sD);
        if (sE < n) recE = load(sE);
        if (sF < n) recF = load(sF);
        if (sG < n) recG = load(sG);
        if (sH < n) recH = load(sH);
        if (sI < n) recI = load(sI);
        stage(recA, s);
        if (sB < n) stage(recB, sB);
        if (sC < n) stage(recC, sC);
        if (sD < n) stage(recD, sD);
        if (sE < n) stage(recE, sE);
        if (sF < n) stage(recF, sF);
        if (sG < n) stage(recG, sG);
        if (sH < n) stage(recH, sH);
        if (sI < n) stage(recI, sI);
    }
#else
    for (u32 s = tid; s < n; s += 2 * blockDim.x) {
        const uint4 recA = load(s);
        const u32 sB = s + blockDim.x;
        uint4 recB;
        if (sB < n)
            recB = load(sB);
        stage(recA, s);
        if (sB < n)
            stage(recB, sB);
    }
#endif
    __syncthreads();

#ifdef EQ_PHASE_TIMING
    const u32 eq_t1 = u32(clock64());
#endif

    const u32 selected_count = min(selected, SELECT_CAPACITY);
    for (u32 pos1 = tid; pos1 < selected_count; pos1 += blockDim.x) {
        u64 meta1 = 0;
        if constexpr (ROUND == 4)
            meta1 = (u64(hashes[pos1 * IN_WORDS] >> 22) << 28) |
                    (hashes[pos1 * IN_WORDS + 1] & 0x0fffffffu);
        const u32 index1 = ROUND == 1 ? 0 : base +
                           (ROUND == 4 ? (u32(meta1) & 0x3fff) : slots[pos1]);

        for (u32 pos0 = next[pos1]; pos0 != 0xffffu; pos0 = next[pos0]) {
            u64 meta0 = 0;
            if constexpr (ROUND == 4)
                meta0 = (u64(hashes[pos0 * IN_WORDS] >> 22) << 28) |
                        (hashes[pos0 * IN_WORDS + 1] & 0x0fffffffu);
            const u32 index0 = ROUND == 1 ? 0 : base +
                               (ROUND == 4 ? (u32(meta0) & 0x3fff) : slots[pos0]);
            // A full tail-word equality is the same cheap cycle rejection
            // used by the original solver, without treating padding as hash.
            if (hashes[pos0 * IN_WORDS + IN_WORDS - 2] ==
                hashes[pos1 * IN_WORDS + IN_WORDS - 2])
                continue;

            u32 x[IN_WORDS];
#pragma unroll
            for (u32 w = 0; w < IN_WORDS; ++w)
                x[w] = hashes[pos0 * IN_WORDS + w] ^
                       hashes[pos1 * IN_WORDS + w];

            const u32 out_bucket = (x[0] >> (32 - SHIFT)) & ((1u << OUT_BUCKET_BITS) - 1);
            const u32 out_slot = atomicAdd(&out_counts[out_bucket], 1);
            if (out_slot >= OUT_CAPACITY)
                continue;

            const u32 out_index = out_bucket * OUT_CAPACITY + out_slot;
            u32 dropped;
            if constexpr (ROUND == 1)
                dropped = ((x[3] & 0x1fffff) << 3) | (x[0] >> 29);
            else if constexpr (ROUND == 2)
                dropped = ((x[0] >> 20) << 12) | (x[IN_WORDS - 1] & 0xfff);
            else if constexpr (ROUND == 3)
                dropped = ((drop_hi[pos0] ^ drop_hi[pos1]) << 16) |
                          ((x[IN_WORDS - 1] & 0xf) << 12) | (x[0] >> 20);
            else if constexpr (ROUND == 4)
                dropped = u32((meta0 ^ meta1) >> 14);
            u32 values[OUT_WORDS];
#pragma unroll
            for (u32 w = 0; w < OUT_WORDS; ++w) {
                const u32 hi = x[w] << SHIFT;
                const u32 lo = w + 1 < IN_WORDS ? x[w + 1] >> (32 - SHIFT) : 0;
                u32 value = hi | lo;
                if (w == OUT_WORDS - 1)
                    value &= LAST_MASK;
                if constexpr (ROUND == 3)
                    if (w == OUT_WORDS - 1)
                        value |= dropped;
                if constexpr (ROUND == 4)
                    if (w == OUT_WORDS - 1)
                        value |= dropped & 0xfffff;
                values[w] = value;
            }

            constexpr u32 SLOT_BITS = IN_BUCKET_BITS == 12 ? 14 : 13;
            constexpr u32 META_SHIFT = 2 * SLOT_BITS + IN_BUCKET_BITS;
            const u32 slot0 = ROUND == 1 ? slots[pos0] : index0 - base;
            const u32 slot1 = ROUND == 1 ? slots[pos1] : index1 - base;
            const u64 parent = u64(slot0) | (u64(slot1) << SLOT_BITS) |
                               (u64(bucket) << (2 * SLOT_BITS)) |
                               (u64(dropped) << META_SHIFT);
            if constexpr (ROUND == 1) {
                // The low 12 hash-padding bits and one more word carry the
                // dropped value; the final 39 bits retain both slots/bucket.
                static_assert(OUT_WORDS == 3, "packed 20-byte record");
                u32 *record = out + out_index * 5;
                record[0] = values[0];
                record[1] = values[1];
                record[2] = values[2] | (dropped >> 12);
                record[3] = ((dropped & 0xfffu) << 20) | (u32(parent) & 0xfffffu);
                record[4] = u32(parent >> 20) & 0x7ffffu;
            } else if constexpr (ROUND == 2 || ROUND == 3) {
                static_assert(OUT_WORDS == 2, "interleaved record");
                reinterpret_cast<uint4 *>(out)[out_index] = make_uint4(
                    values[0], values[1], u32(parent), u32(parent >> 32));
            } else {
                static_assert(OUT_WORDS == 1, "packed final record");
                u32 *record = out + out_index * 3;
                record[0] = values[0];
                record[1] = u32(parent >> 32);
                record[2] = u32(parent);
            }
        }
    }

#ifdef EQ_PHASE_TIMING
    __syncthreads();
    if (tid == 0) {
        atomicAdd(&eq_phase_cycles[ROUND][0], (unsigned long long)u32(eq_t1 - eq_t0));
        atomicAdd(&eq_phase_cycles[ROUND][1], (unsigned long long)u32(u32(clock64()) - eq_t1));
        atomicAdd(&eq_phase_cycles[ROUND][2], 1ull);
    }
#endif
}

__device__ __forceinline__ void bb_order(u32 *indices, u32 half) {
    if (indices[0] <= indices[half])
        return;
    for (u32 i = 0; i < half; ++i) {
        const u32 tmp = indices[i];
        indices[i] = indices[half + i];
        indices[half + i] = tmp;
    }
}

__device__ __forceinline__ void bb_expand1(const u32 *leaves, const u32 *p1,
                                           u32 index, u32 *out) {
    const u64 p = (p1[index * 5 + 3] & 0xfffffu) |
                  (u64(p1[index * 5 + 4] & 0x7ffffu) << 20);
    const u32 bucket = (p >> 26) & (BB_MID_BUCKETS - 1);
    out[0] = bb_leaf_get(leaves, bucket * BB_MID_CAPACITY + (u32(p) & 0x1fff), 4) & BB_LEAF_MASK;
    out[1] = bb_leaf_get(leaves, bucket * BB_MID_CAPACITY + ((p >> 13) & 0x1fff), 4) & BB_LEAF_MASK;
    bb_order(out, 1);
}

__device__ __forceinline__ void bb_expand2(const u32 *leaves, const u32 *p1,
                                           const u64 *p2, u32 index, u32 *out) {
    const u64 p = p2[index * 2 + 1];
    const u32 bucket = (p >> 28) & (BB_BUCKETS - 1);
    bb_expand1(leaves, p1, bucket * BB_CAPACITY + (u32(p) & 0x3fff), out);
    bb_expand1(leaves, p1, bucket * BB_CAPACITY + ((p >> 14) & 0x3fff), out + 2);
    bb_order(out, 2);
}

__device__ __forceinline__ void bb_expand3(const u32 *leaves, const u32 *p1,
                                           const u64 *p2, const u64 *p3,
                                           u32 index, u32 *out) {
    const u64 p = p3[index * 2 + 1];
    const u32 bucket = (p >> 28) & (BB_BUCKETS - 1);
    bb_expand2(leaves, p1, p2, bucket * BB_CAPACITY + (u32(p) & 0x3fff), out);
    bb_expand2(leaves, p1, p2, bucket * BB_CAPACITY + ((p >> 14) & 0x3fff), out + 4);
    bb_order(out, 4);
}

// Round-4 records are (value, parent hi, parent lo).
__device__ __forceinline__ u64 bb_r4_parent(const u32 *r4, u32 index) {
    return (u64(r4[index * 3 + 1]) << 32) | r4[index * 3 + 2];
}

__device__ __forceinline__ void bb_expand4(const u32 *leaves, const u32 *p1,
                                           const u64 *p2, const u64 *p3,
                                           const u32 *r4, u32 index, u32 *out) {
    const u64 p = bb_r4_parent(r4, index);
    const u32 bucket = (p >> 28) & (BB_BUCKETS - 1);
    bb_expand3(leaves, p1, p2, p3, bucket * BB_CAPACITY + (u32(p) & 0x3fff), out);
    bb_expand3(leaves, p1, p2, p3, bucket * BB_CAPACITY + ((p >> 14) & 0x3fff), out + 8);
    bb_order(out, 8);
}

__device__ __forceinline__ void bb_candidate(equi *eq, const u32 *leaves,
                                              const u32 *p1, const u64 *p2,
                                              const u64 *p3, const u32 *r4,
                                              u32 index0, u32 index1) {
    u32 indices[PROOFSIZE];
    bb_expand4(leaves, p1, p2, p3, r4, index0, indices);
    bb_expand4(leaves, p1, p2, p3, r4, index1, indices + 16);
    bb_order(indices, 16);

    // Do not let cyclic candidates occupy the small result buffer.
    for (u32 i = 0; i < PROOFSIZE; ++i)
        for (u32 j = i + 1; j < PROOFSIZE; ++j)
            if (indices[i] == indices[j])
                return;

    const u32 soli = atomicAdd(&eq->nsols, 1);
    if (soli < MAXSOLS)
        memcpy(eq->sols[soli], indices, sizeof(indices));
}

static constexpr u32 BB_MAX_CANDIDATES = 256;

__global__ __launch_bounds__(512) void bb_final_candidates(
    const u32 *__restrict__ in, const u32 *__restrict__ counts,
    u64 *__restrict__ candidates, u32 *__restrict__ candidate_count) {
    constexpr u32 SELECT_CAPACITY = 4592;
    constexpr u32 LINK_MASK = 0x1fff;
    __shared__ u32 heads[2048];
    __shared__ u32 selected;
    __shared__ u32 hashes[SELECT_CAPACITY];
    // slot[13:0], previous selected position[26:14], tail high bits[30:27]
    __shared__ u32 meta[SELECT_CAPACITY];

    const u32 part = blockIdx.x & 1;
    const u32 bucket = blockIdx.x >> 1;
    const u32 tid = threadIdx.x;
    const u32 n = min(counts[bucket], BB_CAPACITY);
    const u32 base = bucket * BB_CAPACITY;
    auto load = [&](u32 s) {
        const u32 *record = in + (base + s) * 3;
        return make_uint2(record[0], record[1]);
    };

    for (u32 k = tid; k < 2048; k += blockDim.x)
        heads[k] = ~0u;
    if (tid == 0)
        selected = 0;
    __syncthreads();

    auto stage = [&](const uint2 rec, const u32 s) {
        const u32 key = rec.x >> 20;
        const bool matched = (key >> 11) == part;
#if defined(__HIP_PLATFORM_AMD__)
        const u32 votes = u32(__ballot(matched));
        if (!votes)
            return;
        const u32 lane = __lane_id();
        const u32 leader = __ffs(votes) - 1;
        u32 pos = __mbcnt_lo(u32(votes), 0);
        u32 base_pos = 0;
        if (lane == leader)
            base_pos = atomicAdd(&selected, __popc(votes));
        pos += __shfl(base_pos, leader);
        if (!matched || pos >= SELECT_CAPACITY)
            return;
#else
        if (!matched)
            return;
        const u32 pos = atomicAdd(&selected, 1);
        if (pos >= SELECT_CAPACITY)
            return;
#endif
        const u32 previous = atomicExch(&heads[key & 2047], pos);
        hashes[pos] = rec.x;
        meta[pos] = s | ((previous & LINK_MASK) << 14) |
                    ((rec.y >> 28) << 27);
    };

    for (u32 s = tid; s < n; s += 17 * blockDim.x) {
        const uint2 recA = load(s);
        const u32 sB = s + blockDim.x;
        const u32 sC = sB + blockDim.x;
        const u32 sD = sC + blockDim.x;
        const u32 sE = sD + blockDim.x;
        const u32 sF = sE + blockDim.x;
        const u32 sG = sF + blockDim.x;
        const u32 sH = sG + blockDim.x;
        const u32 sI = sH + blockDim.x;
        const u32 sJ = sI + blockDim.x;
        const u32 sK = sJ + blockDim.x;
        const u32 sL = sK + blockDim.x;
        const u32 sM = sL + blockDim.x;
        const u32 sN = sM + blockDim.x;
        const u32 sO = sN + blockDim.x;
        const u32 sP = sO + blockDim.x;
        const u32 sQ = sP + blockDim.x;
        uint2 recB, recC, recD, recE, recF, recG, recH, recI, recJ, recK, recL,
              recM, recN, recO, recP, recQ;
        if (sB < n) recB = load(sB);
        if (sC < n) recC = load(sC);
        if (sD < n) recD = load(sD);
        if (sE < n) recE = load(sE);
        if (sF < n) recF = load(sF);
        if (sG < n) recG = load(sG);
        if (sH < n) recH = load(sH);
        if (sI < n) recI = load(sI);
        if (sJ < n) recJ = load(sJ);
        if (sK < n) recK = load(sK);
        if (sL < n) recL = load(sL);
        if (sM < n) recM = load(sM);
        if (sN < n) recN = load(sN);
        if (sO < n) recO = load(sO);
        if (sP < n) recP = load(sP);
        if (sQ < n) recQ = load(sQ);
        stage(recA, s);
        if (sB < n) stage(recB, sB);
        if (sC < n) stage(recC, sC);
        if (sD < n) stage(recD, sD);
        if (sE < n) stage(recE, sE);
        if (sF < n) stage(recF, sF);
        if (sG < n) stage(recG, sG);
        if (sH < n) stage(recH, sH);
        if (sI < n) stage(recI, sI);
        if (sJ < n) stage(recJ, sJ);
        if (sK < n) stage(recK, sK);
        if (sL < n) stage(recL, sL);
        if (sM < n) stage(recM, sM);
        if (sN < n) stage(recN, sN);
        if (sO < n) stage(recO, sO);
        if (sP < n) stage(recP, sP);
        if (sQ < n) stage(recQ, sQ);
    }
    __syncthreads();

    const u32 selected_count = min(selected, SELECT_CAPACITY);
    for (u32 pos1 = tid; pos1 < selected_count; pos1 += blockDim.x) {
        const u32 hash1 = hashes[pos1];
        const u32 meta1 = meta[pos1];
        for (u32 pos0 = (meta1 >> 14) & LINK_MASK; pos0 != LINK_MASK;
             pos0 = (meta[pos0] >> 14) & LINK_MASK) {
            if (hashes[pos0] != hash1 || ((meta[pos0] ^ meta1) >> 27) != 0)
                continue;
            const u32 candidate_pos = atomicAdd(candidate_count, 1);
            if (candidate_pos < BB_MAX_CANDIDATES)
                candidates[candidate_pos] = u64(base + (meta[pos0] & 0x3fff)) |
                                            (u64(base + (meta1 & 0x3fff)) << 32);
        }
    }
}
__global__ __launch_bounds__(256) void bb_combine_candidates(
    equi *eq, const u32 *leaves, const u32 *p1, const u64 *p2,
    const u64 *p3, const u32 *r4, const u64 *candidates,
    const u32 *candidate_count) {
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= min(*candidate_count, BB_MAX_CANDIDATES))
        return;
    const u64 candidate = candidates[id];
    bb_candidate(eq, leaves, p1, p2, p3, r4, u32(candidate), u32(candidate >> 32));
}
