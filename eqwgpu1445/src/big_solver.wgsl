// Large-bucket Equihash (144,5) solver. One workgroup cooperates on each
// bucket partition, avoiding the old solver's per-invocation 64-slot arrays.

const BIG_BUCKETS: u32 = 4096u;
const BIG_CAPACITY: u32 = 8688u;
const BIG_SLOTS: u32 = BIG_BUCKETS * BIG_CAPACITY;
const MID_BUCKETS: u32 = 8192u;
const MID_CAPACITY: u32 = 4592u;
const NBLOCKS: u32 = 11184811u;
const MAXSOLS: u32 = 10u;
const LINK_NIL: u32 = 4095u;

override NTHREADS: u32 = 1048576u;
override ROUND: u32 = 1u;

struct Params {
    h: array<u32, 16>,
    rem: array<u32, 13>,
}

struct Sols {
    n: atomic<u32>,
    idx: array<u32, 320>,
}

@group(0) @binding(0) var<storage, read> params: Params;
@group(0) @binding(1) var<storage, read_write> big0: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> big1: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> big2: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> leaves: array<u32>;
@group(0) @binding(5) var<storage, read_write> parents1: array<vec2<u32>>;
@group(0) @binding(6) var<storage, read_write> counts0: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> counts1: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> counts2: array<atomic<u32>>;
@group(0) @binding(9) var<storage, read_write> counts3: array<atomic<u32>>;
@group(0) @binding(10) var<storage, read_write> counts4: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read_write> sols: Sols;

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
fn ror63v(a: vec2<u32>) -> vec2<u32> {
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

var<private> digest: array<u32, 14>;

fn blake2b_block_hash(idx: u32) {
    for (var k = 0u; k < 16u; k++) { m[k] = vec2<u32>(0u, 0u); }
    for (var i = 0u; i < 13u; i++) {
        if ((i % 2u) == 0u) { m[i / 2u].x = params.rem[i]; } else { m[i / 2u].y = params.rem[i]; }
    }
    m[6u].y = idx;
    for (var i = 0u; i < 8u; i++) {
        v[i] = vec2<u32>(params.h[2u * i], params.h[2u * i + 1u]);
    }
    v[8u] = IV[0]; v[9u] = IV[1]; v[10u] = IV[2]; v[11u] = IV[3];
    v[12u] = IV[4] ^ vec2<u32>(184u, 0u);
    v[13u] = IV[5];
    v[14u] = ~IV[6];
    v[15u] = IV[7];
    blake_round(0u);
    blake_round(1u);
    blake_round(2u);
    blake_round(3u);
    blake_round(4u);
    blake_round(5u);
    blake_round(6u);
    blake_round(7u);
    blake_round(8u);
    blake_round(9u);
    blake_round(10u);
    blake_round(11u);
    for (var i = 0u; i < 7u; i++) {
        let h = vec2<u32>(params.h[2u * i], params.h[2u * i + 1u]) ^ v[i] ^ v[i + 8u];
        digest[2u * i] = h.x;
        digest[2u * i + 1u] = h.y;
    }
}

fn dbyte(b: u32) -> u32 {
    return (digest[b / 4u] >> (8u * (b % 4u))) & 0xffu;
}

fn be_word(base: u32) -> u32 {
    return (dbyte(base) << 24u) | (dbyte(base + 1u) << 16u) |
        (dbyte(base + 2u) << 8u) | dbyte(base + 3u);
}

@compute @workgroup_size(256)
fn digitH(@builtin(global_invocation_id) gid: vec3<u32>) {
    for (var block = gid.x; block < NBLOCKS; block += NTHREADS) {
        blake2b_block_hash(block);
        for (var i = 0u; i < 3u; i++) {
            let leaf = block * 3u + i;
            if (leaf >= 33554432u) { break; }
            let base = i * 18u;
            let a0 = be_word(base);
            let bucket = a0 >> 20u;
            let slot = atomicAdd(&counts0[bucket], 1u);
            if (slot >= BIG_CAPACITY) { continue; }
            let a1 = be_word(base + 4u);
            let a2 = be_word(base + 8u);
            let a3 = be_word(base + 12u);
            let a4 = (dbyte(base + 16u) << 24u) | (dbyte(base + 17u) << 16u);
            let index = bucket * BIG_CAPACITY + slot;
            big0[index] = vec4<u32>((a0 << 12u) | (a1 >> 20u),
                (a1 << 12u) | (a2 >> 20u), (a2 << 12u) | (a3 >> 20u),
                (a3 << 12u) | (a4 >> 20u));
            leaves[index] = (a4 << 12u) | leaf;
        }
    }
}

// ---------------- cooperative collision rounds ----------------

var<workgroup> heads: array<atomic<u32>, 1024>;
var<workgroup> selected: atomic<u32>;
// low 12 bits: previous selected position, remaining bits: original slot
var<workgroup> links: array<u32, 2816>;
var<workgroup> staged_hashes: array<u32, 11264>;

fn staged_hash(pos: u32) -> vec4<u32> {
    let o = pos * 4u;
    return vec4<u32>(staged_hashes[o], staged_hashes[o + 1u],
        staged_hashes[o + 2u], staged_hashes[o + 3u]);
}

fn round_count(r: u32, bucket: u32) -> u32 {
    switch r {
        case 1u: { return min(atomicLoad(&counts0[bucket]), BIG_CAPACITY); }
        case 2u: { return min(atomicLoad(&counts1[bucket]), MID_CAPACITY); }
        case 3u: { return min(atomicLoad(&counts2[bucket]), BIG_CAPACITY); }
        default: { return min(atomicLoad(&counts3[bucket]), MID_CAPACITY); }
    }
}

fn round_record(r: u32, index: u32) -> vec4<u32> {
    switch r {
        case 1u, 3u: { return big0[index]; }
        default: { return big1[index]; }
    }
}

fn round_hash(r: u32, rec: vec4<u32>, index: u32, slot: u32) -> vec4<u32> {
    switch r {
        case 1u: {
            return vec4<u32>((rec.x & 0x000fffffu) | (leaves[index] & 0xf0000000u),
                rec.y, rec.z, rec.w);
        }
        case 2u: {
            return vec4<u32>((rec.x & 0x001fffffu) | ((rec.w >> 13u) << 21u),
                rec.y, rec.z | (rec.w & 0x1fffu), 0u);
        }
        case 3u: {
            let dropped = rec.w >> 7u;
            return vec4<u32>((rec.x & 0x000fffffu) | ((dropped & 0xfffu) << 20u),
                rec.y | ((dropped >> 12u) & 0xfu), dropped >> 16u, 0u);
        }
        default: {
            let tail = rec.y & 0xffffffu;
            let meta_lo = slot | ((tail & 0x7ffffu) << 13u);
            let meta_hi = tail >> 19u;
            return vec4<u32>((rec.x & 0x00ffffffu) | (((meta_lo >> 29u) | (meta_hi << 3u)) << 24u),
                (rec.y & 0xe0000000u) | (meta_lo & 0x1fffffffu), meta_lo, meta_hi);
        }
    }
}

fn parent14(bucket: u32, slot0: u32, slot1: u32, dropped: u32) -> vec2<u32> {
    return vec2<u32>(slot0 | (slot1 << 14u) | ((bucket & 0xfu) << 28u),
        (bucket >> 4u) | (dropped << 8u));
}

fn parent13(bucket: u32, slot0: u32, slot1: u32, dropped: u32) -> vec2<u32> {
    return vec2<u32>(slot0 | (slot1 << 13u) | ((bucket & 0x3fu) << 26u),
        (bucket >> 6u) | (dropped << 7u));
}

fn emit_pair(r: u32, bucket: u32, slot0: u32, slot1: u32, h0: vec4<u32>, h1: vec4<u32>) {
    let x = h0 ^ h1;
    switch r {
        case 1u: {
            let out_bucket = (x.x >> 7u) & 0x1fffu;
            let out_slot = atomicAdd(&counts1[out_bucket], 1u);
            if (out_slot >= MID_CAPACITY) { return; }
            let index = out_bucket * MID_CAPACITY + out_slot;
            let dropped = ((x.w & 0xfffffu) << 4u) | (x.x >> 28u);
            big1[index] = vec4<u32>((x.x << 25u) | (x.y >> 7u),
                (x.y << 25u) | (x.z >> 7u),
                ((x.z << 25u) | (x.w >> 7u)) & 0xffffe000u, dropped);
            let p = parent14(bucket, slot0, slot1, dropped);
            parents1[index] = p;
        }
        case 2u: {
            let out_bucket = (x.x >> 9u) & 0xfffu;
            let out_slot = atomicAdd(&counts2[out_bucket], 1u);
            if (out_slot >= BIG_CAPACITY) { return; }
            let index = out_bucket * BIG_CAPACITY + out_slot;
            let dropped = ((x.x >> 21u) << 13u) | (x.z & 0x1fffu);
            let p = parent13(bucket, slot0, slot1, dropped);
            big0[index] = vec4<u32>((x.x << 23u) | (x.y >> 9u),
                ((x.y << 23u) | (x.z >> 9u)) & 0xfffffff0u, p.x, p.y);
        }
        case 3u: {
            let out_bucket = (x.x >> 7u) & 0x1fffu;
            let out_slot = atomicAdd(&counts3[out_bucket], 1u);
            if (out_slot >= MID_CAPACITY) { return; }
            let index = out_bucket * MID_CAPACITY + out_slot;
            let dropped = ((h0.z ^ h1.z) << 16u) | ((x.y & 0xfu) << 12u) | (x.x >> 20u);
            let p = parent14(bucket, slot0, slot1, dropped);
            big1[index] = vec4<u32>((x.x << 25u) | (x.y >> 7u),
                ((x.y << 25u) & 0xe0000000u) | dropped, p.x, p.y);
        }
        default: {
            let out_bucket = (x.x >> 9u) & 0xfffu;
            let out_slot = atomicAdd(&counts4[out_bucket], 1u);
            if (out_slot >= BIG_CAPACITY) { return; }
            let index = out_bucket * BIG_CAPACITY + out_slot;
            let dropped = ((h0.z ^ h1.z) >> 13u) | ((h0.w ^ h1.w) << 19u);
            let p = parent13(bucket, slot0, slot1, dropped);
            big2[index] = vec4<u32>(
                (((x.x << 23u) | (x.y >> 9u)) & 0xfff00000u) | (dropped & 0xfffffu),
                p.x, p.y, 0u);
        }
    }
}

@compute @workgroup_size(1024)
fn digitR(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let r = ROUND;
    let big_round = r == 1u || r == 3u;
    let part_bits = select(1u, 2u, big_round);
    let part_mask = (1u << part_bits) - 1u;
    let part = wid.x & part_mask;
    let bucket = wid.x >> part_bits;
    let input_capacity = select(MID_CAPACITY, BIG_CAPACITY, big_round);
    let select_capacity = select(2816u, 2240u, big_round);
    let n = round_count(r, bucket);
    let base = bucket * input_capacity;

    atomicStore(&heads[tid], 0xffffffffu);
    if (tid == 0u) { atomicStore(&selected, 0u); }
    workgroupBarrier();

    for (var slot = tid; slot < n; slot += 1024u) {
        let rec = round_record(r, base + slot);
        let key = rec.x >> select(21u, 20u, big_round);
        if ((key >> 10u) != part) { continue; }
        let pos = atomicAdd(&selected, 1u);
        if (pos >= select_capacity) { continue; }
        let h = round_hash(r, rec, base + slot, slot);
        staged_hashes[pos * 4u] = h.x;
        staged_hashes[pos * 4u + 1u] = h.y;
        staged_hashes[pos * 4u + 2u] = h.z;
        staged_hashes[pos * 4u + 3u] = h.w;
        let previous = atomicExchange(&heads[key & 1023u], pos);
        links[pos] = (slot << 12u) | (previous & LINK_NIL);
    }
    workgroupBarrier();

    let selected_count = min(atomicLoad(&selected), select_capacity);
    for (var pos1 = tid; pos1 < selected_count; pos1 += 1024u) {
        let link1 = links[pos1];
        let slot1 = link1 >> 12u;
        let h1 = staged_hash(pos1);
        var pos0 = link1 & LINK_NIL;
        loop {
            if (pos0 == LINK_NIL) { break; }
            let link0 = links[pos0];
            let slot0 = link0 >> 12u;
            let h0 = staged_hash(pos0);
            var same_tail = h0.x == h1.x;
            if (r == 1u) { same_tail = h0.z == h1.z; }
            if (r == 2u) { same_tail = h0.y == h1.y; }
            if (!same_tail) { emit_pair(r, bucket, slot0, slot1, h0, h1); }
            pos0 = link0 & LINK_NIL;
        }
    }
}

// ---------------- final collision and tree expansion ----------------

var<private> indices: array<u32, 32>;

fn order_indices(offset: u32, half: u32) {
    if (indices[offset] <= indices[offset + half]) { return; }
    for (var i = 0u; i < half; i++) {
        let t = indices[offset + i];
        indices[offset + i] = indices[offset + half + i];
        indices[offset + half + i] = t;
    }
}

fn parent14_bucket(p: vec2<u32>) -> u32 { return (p.x >> 28u) | ((p.y & 0xffu) << 4u); }
fn parent13_bucket(p: vec2<u32>) -> u32 { return (p.x >> 26u) | ((p.y & 0x7fu) << 6u); }

fn expand1(index: u32, out: u32) {
    let p = parents1[index];
    let base = parent14_bucket(p) * BIG_CAPACITY;
    indices[out] = leaves[base + (p.x & 0x3fffu)] & 0x1ffffffu;
    indices[out + 1u] = leaves[base + ((p.x >> 14u) & 0x3fffu)] & 0x1ffffffu;
    order_indices(out, 1u);
}

fn expand2(index: u32, out: u32) {
    let rec = big0[index];
    let p = rec.zw;
    let base = parent13_bucket(p) * MID_CAPACITY;
    expand1(base + (p.x & 0x1fffu), out);
    expand1(base + ((p.x >> 13u) & 0x1fffu), out + 2u);
    order_indices(out, 2u);
}

fn expand3(index: u32, out: u32) {
    let rec = big1[index];
    let p = rec.zw;
    let base = parent14_bucket(p) * BIG_CAPACITY;
    expand2(base + (p.x & 0x3fffu), out);
    expand2(base + ((p.x >> 14u) & 0x3fffu), out + 4u);
    order_indices(out, 4u);
}

fn expand4(index: u32, out: u32) {
    let rec = big2[index];
    let p = rec.yz;
    let base = parent13_bucket(p) * MID_CAPACITY;
    expand3(base + (p.x & 0x1fffu), out);
    expand3(base + ((p.x >> 13u) & 0x1fffu), out + 8u);
    order_indices(out, 8u);
}

fn candidate(index0: u32, index1: u32) {
    expand4(index0, 0u);
    expand4(index1, 16u);
    order_indices(0u, 16u);
    for (var i = 0u; i < 32u; i++) {
        for (var j = i + 1u; j < 32u; j++) {
            if (indices[i] == indices[j]) { return; }
        }
    }
    let solution = atomicAdd(&sols.n, 1u);
    if (solution >= MAXSOLS) { return; }
    for (var i = 0u; i < 32u; i++) { sols.idx[solution * 32u + i] = indices[i]; }
}

@compute @workgroup_size(1024)
fn digitK(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let part = wid.x & 3u;
    let bucket = wid.x >> 2u;
    let n = min(atomicLoad(&counts4[bucket]), BIG_CAPACITY);
    let base = bucket * BIG_CAPACITY;

    atomicStore(&heads[tid], 0xffffffffu);
    if (tid == 0u) { atomicStore(&selected, 0u); }
    workgroupBarrier();

    for (var slot = tid; slot < n; slot += 1024u) {
        let key = big2[base + slot].x >> 20u;
        if ((key >> 10u) != part) { continue; }
        let pos = atomicAdd(&selected, 1u);
        if (pos >= 2240u) { continue; }
        let previous = atomicExchange(&heads[key & 1023u], pos);
        links[pos] = (slot << 12u) | (previous & LINK_NIL);
    }
    workgroupBarrier();

    let selected_count = min(atomicLoad(&selected), 2240u);
    for (var pos1 = tid; pos1 < selected_count; pos1 += 1024u) {
        let link1 = links[pos1];
        let slot1 = link1 >> 12u;
        let index1 = base + slot1;
        let rec1 = big2[index1];
        var pos0 = link1 & LINK_NIL;
        loop {
            if (pos0 == LINK_NIL) { break; }
            let link0 = links[pos0];
            let slot0 = link0 >> 12u;
            let index0 = base + slot0;
            let rec0 = big2[index0];
            if (rec0.x == rec1.x && (((rec0.z ^ rec1.z) >> 7u) & 0xffffffu) == 0u) {
                candidate(index0, index1);
            }
            pos0 = link0 & LINK_NIL;
        }
    }
}
