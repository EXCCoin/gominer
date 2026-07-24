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
    return word < 4 ? data[index * 4 + word] : data[BB_SLOTS * 4 + index];
}

__device__ __forceinline__ u32 bb_be32(const uchar *p) {
    return (u32(p[0]) << 24) | (u32(p[1]) << 16) | (u32(p[2]) << 8) | p[3];
}

// Bucket by the first 12 bits.  The five stored words contain the remaining
// 132 hash bits, left-aligned; the unused low 25 bits carry the leaf index.
__global__ void bb_digitH(equi *eq, u32 *out, u32 *counts) {
    const u32 id = blockIdx.x * blockDim.x + threadIdx.x;
    blake2b_pre pre;
    blake2b_precompute(&eq->blake_ctx, &pre);
    uchar hash[HASHOUT];

    for (u32 block = id; block < NBLOCKS; block += eq->nthreads) {
        blake2b_gpu_hash_pre(&pre, block, hash);
        u32 buckets[HASHESPERBLAKE];
        u32 slots[HASHESPERBLAKE];
#pragma unroll
        for (u32 i = 0; i < HASHESPERBLAKE; ++i) {
            const u32 leaf = block * HASHESPERBLAKE + i;
            if (leaf >= NHASHES) {
                slots[i] = BB_CAPACITY;
                continue;
            }
            const uchar *h = hash + i * WN / 8;
            const u32 a0 = bb_be32(h);
            buckets[i] = a0 >> 20;
            slots[i] = atomicAdd(&counts[buckets[i]], 1);
        }
#pragma unroll
        for (u32 i = 0; i < HASHESPERBLAKE; ++i) {
            if (slots[i] >= BB_CAPACITY)
                continue;

            const u32 leaf = block * HASHESPERBLAKE + i;
            const uchar *h = hash + i * WN / 8;
            const u32 a0 = bb_be32(h);
            const u32 a1 = bb_be32(h + 4);
            const u32 a2 = bb_be32(h + 8);
            const u32 a3 = bb_be32(h + 12);
            const u32 a4 = (u32(h[16]) << 24) | (u32(h[17]) << 16);
            const u32 index = buckets[i] * BB_CAPACITY + slots[i];
            reinterpret_cast<uint4 *>(out)[index] = make_uint4(
                (a0 << 12) | (a1 >> 20),
                (a1 << 12) | (a2 >> 20),
                (a2 << 12) | (a3 >> 20),
                (a3 << 12) | (a4 >> 20));
            out[BB_SLOTS * 4 + index] = (a4 << 12) | leaf;
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
                                                 u32 *__restrict__ out_counts,
                                                 u64 *__restrict__ parents) {
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
    __shared__ u16 slots[ROUND == 4 ? 1 : SELECT_CAPACITY];
    __shared__ u16 next[SELECT_CAPACITY];
    __shared__ uchar drop_hi[ROUND == 3 ? SELECT_CAPACITY : 1];

    const u32 part = blockIdx.x & (PARTS - 1);
    const u32 bucket = blockIdx.x >> PART_BITS;
    const u32 tid = threadIdx.x;
    const u32 n = min(in_counts[bucket], IN_CAPACITY);
    const u32 base = bucket * IN_CAPACITY;

    for (u32 k = tid; k < KEYS; k += blockDim.x)
        heads[k] = ~0u;
    if (tid == 0)
        selected = 0;
    __syncthreads();

#ifdef EQ_PHASE_TIMING
    const unsigned long long eq_t0 = clock64();
#endif

    // Compact this partition into shared memory once; collision pairs then
    // reuse the staged hashes instead of issuing random global rereads.
    // Every input layer is a 16-byte record, loaded with a single vector
    // load that serves both the partition-key check and the staged data.
    // Two slots per iteration keep two loads in flight (the phase is
    // load-latency bound, not bandwidth bound).
    auto stage = [&](const uint4 rec, const u32 s) {
        const u32 key = rec.x >> (32 - COLLISION_BITS);
        if ((key >> KEY_BITS) != part)
            return;
        const u32 pos = atomicAdd(&selected, 1);
        if (pos >= SELECT_CAPACITY)
            return;
        if constexpr (ROUND < 4)
            slots[pos] = s;
        if constexpr (ROUND == 1) {
            // Word 0's key bits cancel within a chain, so its top nibble is
            // free to carry the 4 tail bits — their pair XOR then falls out
            // of x[0] with no tails[] array.
            hashes[pos * IN_WORDS] = (rec.x & 0x000fffffu) |
                                     (bb_leaf_get(in, base + s, 4) & 0xf0000000u);
            hashes[pos * IN_WORDS + 1] = rec.y;
            hashes[pos * IN_WORDS + 2] = rec.z;
            hashes[pos * IN_WORDS + 3] = rec.w;
        } else if constexpr (ROUND == 2) {
            const u32 dropped = rec.w;
            hashes[pos * IN_WORDS] = (rec.x & 0x001fffffu) | ((dropped >> 13) << 21);
            hashes[pos * IN_WORDS + 1] = rec.y;
            hashes[pos * IN_WORDS + 2] = rec.z | (dropped & 0x1fff);
        } else if constexpr (ROUND == 3) {
            const u32 dropped = u32(((u64(rec.w) << 32) | rec.z) >> 39);
            hashes[pos * IN_WORDS] = (rec.x & 0x000fffffu) | ((dropped & 0xfff) << 20);
            hashes[pos * IN_WORDS + 1] = rec.y | ((dropped >> 12) & 0xf);
            drop_hi[pos] = dropped >> 16;
        } else if constexpr (ROUND == 4) {
            const u64 meta = u64(s) | (u64(rec.y & 0xffffff) << 13);
            hashes[pos * IN_WORDS] = (rec.x & 0x00ffffffu) | (u32(meta >> 29) << 24);
            hashes[pos * IN_WORDS + 1] = (rec.y & 0xe0000000u) | (u32(meta) & 0x1fffffffu);
        }
        next[pos] = (u16)atomicExch(&heads[key & KEY_MASK], pos);
    };
    for (u32 s = tid; s < n; s += 2 * blockDim.x) {
        const uint4 recA = reinterpret_cast<const uint4 *>(in)[base + s];
        const u32 sB = s + blockDim.x;
        uint4 recB;
        if (sB < n)
            recB = reinterpret_cast<const uint4 *>(in)[base + sB];
        stage(recA, s);
        if (sB < n)
            stage(recB, sB);
    }
    __syncthreads();

#ifdef EQ_PHASE_TIMING
    const unsigned long long eq_t1 = clock64();
#endif

    const u32 selected_count = min(selected, SELECT_CAPACITY);
    for (u32 pos1 = tid; pos1 < selected_count; pos1 += blockDim.x) {
        u64 meta1 = 0;
        if constexpr (ROUND == 4)
            meta1 = (u64(hashes[pos1 * IN_WORDS] >> 24) << 29) |
                    (hashes[pos1 * IN_WORDS + 1] & 0x1fffffffu);
        const u32 index1 = ROUND == 1 ? 0 : base +
                           (ROUND == 4 ? (u32(meta1) & 0x1fff) : slots[pos1]);

        for (u32 pos0 = next[pos1]; pos0 != 0xffffu; pos0 = next[pos0]) {
            u64 meta0 = 0;
            if constexpr (ROUND == 4)
                meta0 = (u64(hashes[pos0 * IN_WORDS] >> 24) << 29) |
                        (hashes[pos0 * IN_WORDS + 1] & 0x1fffffffu);
            const u32 index0 = ROUND == 1 ? 0 : base +
                               (ROUND == 4 ? (u32(meta0) & 0x1fff) : slots[pos0]);
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
                dropped = ((x[3] & 0xfffff) << 4) | (x[0] >> 28);
            else if constexpr (ROUND == 2)
                dropped = ((x[0] >> 21) << 13) | (x[IN_WORDS - 1] & 0x1fff);
            else if constexpr (ROUND == 3)
                dropped = ((drop_hi[pos0] ^ drop_hi[pos1]) << 16) |
                          ((x[IN_WORDS - 1] & 0xf) << 12) | (x[0] >> 20);
            else if constexpr (ROUND == 4)
                dropped = u32((meta0 ^ meta1) >> 13);
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
                // 16-byte record; word 3 carries the dropped bits so round 2
                // never has to read the parent plane.
                static_assert(OUT_WORDS == 3, "uint4 record");
                reinterpret_cast<uint4 *>(out)[out_index] = make_uint4(
                    values[0], values[1], values[2], dropped);
                parents[out_index] = parent;
            } else if constexpr (ROUND == 2 || ROUND == 3) {
                static_assert(OUT_WORDS == 2, "interleaved record");
                reinterpret_cast<uint4 *>(out)[out_index] = make_uint4(
                    values[0], values[1], u32(parent), u32(parent >> 32));
            } else {
                // Round 4: one record holds the final-layer word and the
                // parent, so bb_final and expansion read a single plane.
                static_assert(OUT_WORDS == 1, "uint4 record");
                reinterpret_cast<uint4 *>(out)[out_index] = make_uint4(
                    values[0], u32(parent), u32(parent >> 32), 0);
            }
        }
    }

#ifdef EQ_PHASE_TIMING
    __syncthreads();
    if (tid == 0) {
        atomicAdd(&eq_phase_cycles[ROUND][0], eq_t1 - eq_t0);
        atomicAdd(&eq_phase_cycles[ROUND][1], (unsigned long long)(clock64() - eq_t1));
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

__device__ __forceinline__ void bb_expand1(const u32 *leaves, const u64 *p1,
                                           u32 index, u32 *out) {
    const u64 p = p1[index];
    const u32 bucket = (p >> 28) & (BB_BUCKETS - 1);
    out[0] = leaves[BB_SLOTS * 4 + bucket * BB_CAPACITY + (u32(p) & 0x3fff)] & BB_LEAF_MASK;
    out[1] = leaves[BB_SLOTS * 4 + bucket * BB_CAPACITY + ((p >> 14) & 0x3fff)] & BB_LEAF_MASK;
    bb_order(out, 1);
}

__device__ __forceinline__ void bb_expand2(const u32 *leaves, const u64 *p1,
                                           const u64 *p2, u32 index, u32 *out) {
    const u64 p = p2[index * 2 + 1];
    const u32 bucket = (p >> 26) & (BB_MID_BUCKETS - 1);
    bb_expand1(leaves, p1, bucket * BB_MID_CAPACITY + (u32(p) & 0x1fff), out);
    bb_expand1(leaves, p1, bucket * BB_MID_CAPACITY + ((p >> 13) & 0x1fff), out + 2);
    bb_order(out, 2);
}

__device__ __forceinline__ void bb_expand3(const u32 *leaves, const u64 *p1,
                                           const u64 *p2, const u64 *p3,
                                           u32 index, u32 *out) {
    const u64 p = p3[index * 2 + 1];
    const u32 bucket = (p >> 28) & (BB_BUCKETS - 1);
    bb_expand2(leaves, p1, p2, bucket * BB_CAPACITY + (u32(p) & 0x3fff), out);
    bb_expand2(leaves, p1, p2, bucket * BB_CAPACITY + ((p >> 14) & 0x3fff), out + 4);
    bb_order(out, 4);
}

// Round-4 records are uint4 (value, parent lo, parent hi, 0).
__device__ __forceinline__ u64 bb_r4_parent(const u32 *r4, u32 index) {
    return (u64(r4[index * 4 + 2]) << 32) | r4[index * 4 + 1];
}

__device__ __forceinline__ void bb_expand4(const u32 *leaves, const u64 *p1,
                                           const u64 *p2, const u64 *p3,
                                           const u32 *r4, u32 index, u32 *out) {
    const u64 p = bb_r4_parent(r4, index);
    const u32 bucket = (p >> 26) & (BB_MID_BUCKETS - 1);
    bb_expand3(leaves, p1, p2, p3, bucket * BB_MID_CAPACITY + (u32(p) & 0x1fff), out);
    bb_expand3(leaves, p1, p2, p3, bucket * BB_MID_CAPACITY + ((p >> 13) & 0x1fff), out + 8);
    bb_order(out, 8);
}

__device__ __forceinline__ void bb_candidate(equi *eq, const u32 *leaves,
                                              const u64 *p1, const u64 *p2,
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

template <u32 PART_BITS>
__global__ __launch_bounds__(256) void bb_final(equi *eq, const u32 *__restrict__ in,
                                                 const u32 *__restrict__ counts,
                                                 const u32 *leaves,
                                                 const u64 *p1, const u64 *p2,
                                                 const u64 *p3) {
    constexpr u32 PARTS = 1u << PART_BITS;
    constexpr u32 KEY_BITS = BB_BUCKET_BITS - PART_BITS;
    constexpr u32 KEYS = 1u << KEY_BITS;
    constexpr u32 KEY_MASK = KEYS - 1;
    __shared__ u32 heads[KEYS];
    __shared__ u16 next[BB_CAPACITY];

    const u32 part = blockIdx.x & (PARTS - 1);
    const u32 bucket = blockIdx.x >> PART_BITS;
    const u32 tid = threadIdx.x;
    const u32 n = min(counts[bucket], BB_CAPACITY);
    const u32 base = bucket * BB_CAPACITY;

    for (u32 k = tid; k < KEYS; k += blockDim.x)
        heads[k] = ~0u;
    __syncthreads();
    for (u32 s = tid; s < n; s += blockDim.x) {
        const u32 key = in[(base + s) * 4] >> 20;
        if ((key >> KEY_BITS) == part)
            next[s] = (u16)atomicExch(&heads[key & KEY_MASK], s);
    }
    __syncthreads();

    for (u32 s1 = tid; s1 < n; s1 += blockDim.x) {
        const u32 index1 = base + s1;
        const u32 h10 = in[index1 * 4];
        const u32 key = h10 >> 20;
        if ((key >> KEY_BITS) != part)
            continue;
        for (u32 s0 = next[s1]; s0 != 0xffffu; s0 = next[s0]) {
            const u32 index0 = base + s0;
            if (in[index0 * 4] == h10 &&
                (((bb_r4_parent(in, index0) ^ bb_r4_parent(in, index1)) >> 39) & 0xffffff) == 0)
                bb_candidate(eq, leaves, p1, p2, p3, in, index0, index1);
        }
    }
}
