// Equihash (144,5) solver — WGSL port of John Tromp's CUDA solver
// (eqcuda1445/solver_details.cuh), same bucket/slot memory layout.
//
// u64 is emulated as vec2<u32> (x = low word, y = high word).

const NBUCKETS: u32 = 1048576u;  // 1 << (DIGITBITS(24) - RESTBITS(4))
const NSLOTS: u32 = 64u;
const NRESTS: u32 = 16u;
const SLOT_WORDS: u32 = 5u;      // tree attr + 4 hash words
const NBLOCKS: u32 = 11184811u;  // ceil(2^25 / HASHESPERBLAKE(3))
const MAXSOLS: u32 = 10u;
const XNIL: u32 = 0xffffffffu;

// Total worker threads; the stride of every bucket loop.
override NTHREADS: u32 = 1048576u;
// Round number for digit_r (1..4).
override ROUND: u32 = 1u;

// Solver state shared with the host (offsets in u32 words):
//   params[0..15]  blake2b midstate h[8] after absorbing the first 128 header
//                  bytes (u64s as lo,hi pairs)
//   params[16..28] the remaining 52 header bytes
struct Params {
    h: array<u32, 16>,
    rem: array<u32, 13>,
}

struct Sols {
    n: atomic<u32>,
    idx: array<u32, 320>, // MAXSOLS * 32 indices
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read_write> heap0: array<u32>;
@group(0) @binding(2) var<storage, read_write> heap1: array<u32>;
@group(0) @binding(3) var<storage, read_write> nslots: array<atomic<u32>>; // [2][NBUCKETS]
@group(0) @binding(4) var<storage, read_write> sols: Sols;

// Tromp's overlapping-layer heap trick: layer l of heap0 starts at word
// offset l (layers alias almost-fully; earlier rounds' data is dead by the
// time it is overwritten).
fn slot0_base(l: u32, bid: u32, slot: u32) -> u32 {
    return l + (bid * NSLOTS + slot) * SLOT_WORDS;
}
fn slot1_base(l: u32, bid: u32, slot: u32) -> u32 {
    return l + (bid * NSLOTS + slot) * SLOT_WORDS;
}

// byte b (little-endian order) of the 4 hash words stored at word offset base+1
fn h0byte(base: u32, b: u32) -> u32 {
    return (heap0[base + 1u + b / 4u] >> (8u * (b % 4u))) & 0xffu;
}
fn h1byte(base: u32, b: u32) -> u32 {
    return (heap1[base + 1u + b / 4u] >> (8u * (b % 4u))) & 0xffu;
}

// ---------------- blake2b (single final-block compress) ----------------

var<private> SIGMA: array<array<u32, 16>, 12> = array<array<u32, 16>, 12>(
    array<u32, 16>(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u),
    array<u32, 16>(14u, 10u, 4u, 8u, 9u, 15u, 13u, 6u, 1u, 12u, 0u, 2u, 11u, 7u, 5u, 3u),
    array<u32, 16>(11u, 8u, 12u, 0u, 5u, 2u, 15u, 13u, 10u, 14u, 3u, 6u, 7u, 1u, 9u, 4u),
    array<u32, 16>(7u, 9u, 3u, 1u, 13u, 12u, 11u, 14u, 2u, 6u, 5u, 10u, 4u, 0u, 15u, 8u),
    array<u32, 16>(9u, 0u, 5u, 7u, 2u, 4u, 10u, 15u, 14u, 1u, 11u, 12u, 6u, 8u, 3u, 13u),
    array<u32, 16>(2u, 12u, 6u, 10u, 0u, 11u, 8u, 3u, 4u, 13u, 7u, 5u, 15u, 14u, 1u, 9u),
    array<u32, 16>(12u, 5u, 1u, 15u, 14u, 13u, 4u, 10u, 0u, 7u, 6u, 3u, 9u, 2u, 8u, 11u),
    array<u32, 16>(13u, 11u, 7u, 14u, 12u, 1u, 3u, 9u, 5u, 0u, 15u, 4u, 8u, 6u, 2u, 10u),
    array<u32, 16>(6u, 15u, 14u, 9u, 11u, 3u, 0u, 8u, 12u, 2u, 13u, 7u, 1u, 4u, 10u, 5u),
    array<u32, 16>(10u, 2u, 8u, 4u, 7u, 6u, 1u, 5u, 15u, 11u, 9u, 14u, 3u, 12u, 13u, 0u),
    array<u32, 16>(0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u),
    array<u32, 16>(14u, 10u, 4u, 8u, 9u, 15u, 13u, 6u, 1u, 12u, 0u, 2u, 11u, 7u, 5u, 3u),
);

const IV = array<vec2<u32>, 8>(
    vec2<u32>(0xf3bcc908u, 0x6a09e667u),
    vec2<u32>(0x84caa73bu, 0xbb67ae85u),
    vec2<u32>(0xfe94f82bu, 0x3c6ef372u),
    vec2<u32>(0x5f1d36f1u, 0xa54ff53au),
    vec2<u32>(0xade682d1u, 0x510e527fu),
    vec2<u32>(0x2b3e6c1fu, 0x9b05688cu),
    vec2<u32>(0xfb41bd6bu, 0x1f83d9abu),
    vec2<u32>(0x137e2179u, 0x5be0cd19u),
);

fn add64(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}
fn ror32v(a: vec2<u32>) -> vec2<u32> { return vec2<u32>(a.y, a.x); }
fn ror24v(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>((a.x >> 24u) | (a.y << 8u), (a.y >> 24u) | (a.x << 8u));
}
fn ror16v(a: vec2<u32>) -> vec2<u32> {
    return vec2<u32>((a.x >> 16u) | (a.y << 16u), (a.y >> 16u) | (a.x << 16u));
}
fn ror63v(a: vec2<u32>) -> vec2<u32> { // == rol 1
    return vec2<u32>((a.x << 1u) | (a.y >> 31u), (a.y << 1u) | (a.x >> 31u));
}

var<private> v: array<vec2<u32>, 16>;
var<private> m: array<vec2<u32>, 16>;

fn G(r: u32, i: u32, ai: u32, bi: u32, ci: u32, di: u32) {
    v[ai] = add64(add64(v[ai], v[bi]), m[SIGMA[r][2u * i]]);
    v[di] = ror32v(v[di] ^ v[ai]);
    v[ci] = add64(v[ci], v[di]);
    v[bi] = ror24v(v[bi] ^ v[ci]);
    v[ai] = add64(add64(v[ai], v[bi]), m[SIGMA[r][2u * i + 1u]]);
    v[di] = ror16v(v[di] ^ v[ai]);
    v[ci] = add64(v[ci], v[di]);
    v[bi] = ror63v(v[bi] ^ v[ci]);
}

fn blake_round(r: u32) {
    G(r, 0u, 0u, 4u, 8u, 12u);
    G(r, 1u, 1u, 5u, 9u, 13u);
    G(r, 2u, 2u, 6u, 10u, 14u);
    G(r, 3u, 3u, 7u, 11u, 15u);
    G(r, 4u, 0u, 5u, 10u, 15u);
    G(r, 5u, 1u, 6u, 11u, 12u);
    G(r, 6u, 2u, 7u, 8u, 13u);
    G(r, 7u, 3u, 4u, 9u, 14u);
}

// digest words of the 54-byte hash for block index `idx`
var<private> digest: array<u32, 14>;

fn blake2b_block_hash(idx: u32) {
    // final block: 52 remainder bytes || 4-byte LE idx || zero padding
    for (var k = 0u; k < 16u; k++) { m[k] = vec2<u32>(0u, 0u); }
    for (var i = 0u; i < 13u; i++) {
        if ((i % 2u) == 0u) { m[i / 2u].x = params.rem[i]; } else { m[i / 2u].y = params.rem[i]; }
    }
    m[6u].y = idx; // u32 word 13
    for (var i = 0u; i < 8u; i++) {
        v[i] = vec2<u32>(params.h[2u * i], params.h[2u * i + 1u]);
    }
    v[8u] = IV[0]; v[9u] = IV[1]; v[10u] = IV[2]; v[11u] = IV[3];
    v[12u] = IV[4] ^ vec2<u32>(184u, 0u); // counter: 128 header + 52 rem + 4 idx
    v[13u] = IV[5];
    v[14u] = ~IV[6]; // last block
    v[15u] = IV[7];
    for (var r = 0u; r < 12u; r++) { blake_round(r); }
    for (var i = 0u; i < 7u; i++) {
        let hw = vec2<u32>(params.h[2u * i], params.h[2u * i + 1u]) ^ v[i] ^ v[i + 8u];
        digest[2u * i] = hw.x;
        digest[2u * i + 1u] = hw.y;
    }
}

fn dbyte(b: u32) -> u32 {
    return (digest[b / 4u] >> (8u * (b % 4u))) & 0xffu;
}

// ---------------- round 0: hash generation + first bucketing ----------------

@compute @workgroup_size(256)
fn digitH(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    for (var block = id; block < NBLOCKS; block += NTHREADS) {
        blake2b_block_hash(block);
        for (var i = 0u; i < 3u; i++) { // HASHESPERBLAKE
            let base = i * 18u; // WN/8 bytes per hash
            let idx0 = block * 3u + i;
            if (idx0 >= 33554432u) { break; } // NHASHES = 2^25
            let bucketid = (((dbyte(base) << 8u) | dbyte(base + 1u)) << 4u) | (dbyte(base + 2u) >> 4u);
            let slot = atomicAdd(&nslots[bucketid], 1u);
            if (slot >= NSLOTS) { continue; }
            let sb = slot0_base(0u, bucketid, slot);
            heap0[sb] = idx0; // leaf tree = raw index
            // store hash bytes 2..17 (16 bytes, nextbo = 0)
            for (var w = 0u; w < 4u; w++) {
                let b0 = base + 2u + 4u * w;
                heap0[sb + 1u + w] = dbyte(b0) | (dbyte(b0 + 1u) << 8u) |
                    (dbyte(b0 + 2u) << 16u) | (dbyte(b0 + 3u) << 24u);
            }
        }
    }
}

// ---------------- rounds 1..4 ----------------
// layout table (144,5 / BUCKBITS 20 / RESTBITS 4), from htlayout(eq, r):
//   r=1: prev heap0 L0 u4 bo0 | next heap1 L0 u4 | dunits 0
//   r=2: prev heap1 L0 u4 bo3 | next heap0 L1 u3 | dunits 1
//   r=3: prev heap0 L1 u3 bo2 | next heap1 L1 u2 | dunits 1
//   r=4: prev heap1 L1 u2 bo1 | next heap0 L2 u1 | dunits 1

fn prev_layer(r: u32) -> u32 { return (r - 1u) / 2u; }
fn next_layer(r: u32) -> u32 { return r / 2u; }
fn prev_units(r: u32) -> u32 {
    switch r {
        case 1u, 2u: { return 4u; }
        case 3u: { return 3u; }
        default: { return 2u; }
    }
}
fn prev_bo(r: u32) -> u32 {
    switch r {
        case 1u: { return 0u; }
        case 2u: { return 3u; }
        case 3u: { return 2u; }
        default: { return 1u; }
    }
}
fn dunits_of(r: u32) -> u32 { return select(1u, 0u, r == 1u); }

var<private> xhashslots: array<u32, 16>;
var<private> nextxhashslot: array<u32, 64>;

@compute @workgroup_size(256)
fn digitR(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = ROUND;
    let odd = (r & 1u) == 1u;       // odd: heap0 -> heap1, even: heap1 -> heap0
    let pl = prev_layer(r);
    let nl = next_layer(r);
    let punits = prev_units(r);
    let pbo = prev_bo(r);
    let du = dunits_of(r);
    let nsIn = select(NBUCKETS, 0u, odd);  // consumed counter bank offset
    let nsOut = select(0u, NBUCKETS, odd);

    let id = gid.x;
    for (var bucketid = id; bucketid < NBUCKETS; bucketid += NTHREADS) {
        for (var i = 0u; i < NRESTS; i++) { xhashslots[i] = XNIL; }
        for (var i = 0u; i < NSLOTS; i++) { nextxhashslot[i] = XNIL; }
        let bsize = min(atomicExchange(&nslots[nsIn + bucketid], 0u), NSLOTS);
        for (var s1 = 0u; s1 < bsize; s1++) {
            let sb1 = select(slot1_base(pl, bucketid, s1), slot0_base(pl, bucketid, s1), odd);
            // getxhash: hash byte at prevbo, low nibble
            var xh: u32;
            if (odd) { xh = h0byte(sb1, pbo) & 0xfu; } else { xh = h1byte(sb1, pbo) & 0xfu; }
            // collisiondata linked list
            var nextslot = xhashslots[xh];
            nextxhashslot[s1] = nextslot;
            xhashslots[xh] = s1;
            for (; nextslot != XNIL;) {
                let s0 = nextslot;
                nextslot = nextxhashslot[s0];
                let sb0 = select(slot1_base(pl, bucketid, s0), slot0_base(pl, bucketid, s0), odd);
                // equal on last significant word => likely duplicate, skip
                var w0: u32; var w1: u32;
                if (odd) {
                    w0 = heap0[sb0 + 1u + punits - 1u]; w1 = heap0[sb1 + 1u + punits - 1u];
                } else {
                    w0 = heap1[sb0 + 1u + punits - 1u]; w1 = heap1[sb1 + 1u + punits - 1u];
                }
                if (w0 == w1) { continue; }

                var xb1: u32; var xb2: u32; var xb3: u32;
                if (odd) {
                    xb1 = h0byte(sb0, pbo + 1u) ^ h0byte(sb1, pbo + 1u);
                    xb2 = h0byte(sb0, pbo + 2u) ^ h0byte(sb1, pbo + 2u);
                    xb3 = h0byte(sb0, pbo + 3u) ^ h0byte(sb1, pbo + 3u);
                } else {
                    xb1 = h1byte(sb0, pbo + 1u) ^ h1byte(sb1, pbo + 1u);
                    xb2 = h1byte(sb0, pbo + 2u) ^ h1byte(sb1, pbo + 2u);
                    xb3 = h1byte(sb0, pbo + 3u) ^ h1byte(sb1, pbo + 3u);
                }
                let xorbucketid = (((xb1 << 8u) | xb2) << 4u) | (xb3 >> 4u);

                let xorslot = atomicAdd(&nslots[nsOut + xorbucketid], 1u);
                if (xorslot >= NSLOTS) { continue; }
                // packed interior tree node
                let attr = (((bucketid << 6u) | s0) << 6u) | s1;
                if (odd) {
                    let xb = slot1_base(nl, xorbucketid, xorslot);
                    heap1[xb] = attr;
                    for (var i = du; i < punits; i++) {
                        heap1[xb + 1u + i - du] = heap0[sb0 + 1u + i] ^ heap0[sb1 + 1u + i];
                    }
                } else {
                    let xb = slot0_base(nl, xorbucketid, xorslot);
                    heap0[xb] = attr;
                    for (var i = du; i < punits; i++) {
                        heap0[xb + 1u + i - du] = heap1[sb0 + 1u + i] ^ heap1[sb1 + 1u + i];
                    }
                }
            }
        }
    }
}

// ---------------- final round: collision => candidate solution ----------------

var<private> ind: array<u32, 32>;

// expand the tree rooted at (bucketid, s0, s1) over trees0 L2 into 32 leaf
// indices; returns true if the solution is (probably) a dupe
fn expand_candidate(bucketid: u32, cs0: u32, cs1: u32) -> bool {
    // level-order expansion; nodes[i] hold packed attrs, expanded in place
    ind[0] = (((bucketid << 6u) | cs0) << 6u) | cs1;
    var count = 1u;
    // step 0 reads trees0 L2, then trees1 L1, trees0 L1, trees1 L0, trees0 L0
    for (var step = 0u; step < 5u; step++) {
        let use0 = (step % 2u) == 0u; // heap0 on steps 0,2,4
        let layer = (4u - step) / 2u; // 2,1,1,0,0
        var i = count;
        loop {
            if (i == 0u) { break; }
            i--;
            let t = ind[i];
            let bid = t >> 12u;
            let s0 = (t >> 6u) & 63u;
            let s1 = t & 63u;
            var a0: u32; var a1: u32;
            if (use0) {
                a0 = heap0[slot0_base(layer, bid, s0)];
                a1 = heap0[slot0_base(layer, bid, s1)];
            } else {
                a0 = heap1[slot1_base(layer, bid, s0)];
                a1 = heap1[slot1_base(layer, bid, s1)];
            }
            ind[2u * i] = a0;
            ind[2u * i + 1u] = a1;
        }
        count *= 2u;
    }
    // bottom-up ordering (matches the recursive orderindices) + dupe shortcut
    for (var s = 1u; s <= 16u; s *= 2u) {
        for (var base = 0u; base < 32u; base += 2u * s) {
            if (ind[base] == ind[base + s]) { return true; }
            if (ind[base] > ind[base + s]) {
                for (var j = 0u; j < s; j++) {
                    let t = ind[base + j];
                    ind[base + j] = ind[base + s + j];
                    ind[base + s + j] = t;
                }
            }
        }
    }
    return false;
}

@compute @workgroup_size(256)
fn digitK(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    for (var bucketid = id; bucketid < NBUCKETS; bucketid += NTHREADS) {
        for (var i = 0u; i < NRESTS; i++) { xhashslots[i] = XNIL; }
        for (var i = 0u; i < NSLOTS; i++) { nextxhashslot[i] = XNIL; }
        let bsize = min(atomicExchange(&nslots[bucketid], 0u), NSLOTS);
        for (var s1 = 0u; s1 < bsize; s1++) {
            let sb1 = slot0_base(2u, bucketid, s1);
            let xh = h0byte(sb1, 0u) & 0xfu; // prevbo = 0, prevunits = 1
            var nextslot = xhashslots[xh];
            nextxhashslot[s1] = nextslot;
            xhashslots[xh] = s1;
            for (; nextslot != XNIL;) {
                let s0 = nextslot;
                nextslot = nextxhashslot[s0];
                let sb0 = slot0_base(2u, bucketid, s0);
                if (heap0[sb0 + 1u] != heap0[sb1 + 1u]) { continue; }
                // prob_disjoint on the two packed attrs
                let a0 = heap0[sb0];
                let a1 = heap0[sb1];
                let xort = a0 ^ a1;
                let disjoint = ((xort >> 12u) != 0u) || (((xort >> 6u) & 63u) != 0u && (xort & 63u) != 0u);
                if (!disjoint) { continue; }
                if (expand_candidate(bucketid, s0, s1)) { continue; }
                let soli = atomicAdd(&sols.n, 1u);
                if (soli < MAXSOLS) {
                    for (var j = 0u; j < 32u; j++) {
                        sols.idx[soli * 32u + j] = ind[j];
                    }
                }
            }
        }
    }
}
