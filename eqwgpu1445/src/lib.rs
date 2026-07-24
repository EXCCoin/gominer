// Portable Equihash (144,5) GPU solver — wgpu host for the WGSL port of
// John Tromp's solver. Exposes the same C ABI as gominer's eqcuda1445, so the
// Go host links either backend unchanged.

use std::cell::Cell;
use std::ffi::c_void;
use std::sync::OnceLock;

const NBUCKETS: u64 = 1 << 20;
const NSLOTS: u64 = 64;
const SLOT_WORDS: u64 = 5;
const MAXSOLS: usize = 10;
const PROOFSIZE: usize = 32;
const HEADER_LEN: usize = 180; // algo v1 Equihash input
const NONCE_OFFSET: usize = 140;
const COMPRESSED_SOL_SIZE: usize = 100;
const WORKGROUP: u32 = 256;
const DEFAULT_NTHREADS: u32 = 1 << 20;

// heap word count: layer offset (max 2) + all slots + one full slot of slack
const HEAP_WORDS: u64 = NBUCKETS * NSLOTS * SLOT_WORDS + 8;

// ---------------- blake2b host side (u64 native) ----------------

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

fn blake2b_compress(h: &mut [u64; 8], block: &[u8], counter: u64, last: bool) {
    let mut m = [0u64; 16];
    for (i, w) in m.iter_mut().enumerate() {
        *w = u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= counter;
    if last {
        v[14] = !v[14];
    }
    macro_rules! g {
        ($r:expr, $i:expr, $a:expr, $b:expr, $c:expr, $d:expr) => {
            v[$a] = v[$a].wrapping_add(v[$b]).wrapping_add(m[SIGMA[$r][2 * $i]]);
            v[$d] = (v[$d] ^ v[$a]).rotate_right(32);
            v[$c] = v[$c].wrapping_add(v[$d]);
            v[$b] = (v[$b] ^ v[$c]).rotate_right(24);
            v[$a] = v[$a].wrapping_add(v[$b]).wrapping_add(m[SIGMA[$r][2 * $i + 1]]);
            v[$d] = (v[$d] ^ v[$a]).rotate_right(16);
            v[$c] = v[$c].wrapping_add(v[$d]);
            v[$b] = (v[$b] ^ v[$c]).rotate_right(63);
        };
    }
    for r in 0..12 {
        g!(r, 0, 0, 4, 8, 12);
        g!(r, 1, 1, 5, 9, 13);
        g!(r, 2, 2, 6, 10, 14);
        g!(r, 3, 3, 7, 11, 15);
        g!(r, 4, 0, 5, 10, 15);
        g!(r, 5, 1, 6, 11, 12);
        g!(r, 6, 2, 7, 8, 13);
        g!(r, 7, 3, 4, 9, 14);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// blake2b state with the Equihash personalization after absorbing the first
/// 128 header bytes: (h, 52-byte remainder).
fn equihash_midstate(header: &[u8; HEADER_LEN]) -> ([u64; 8], [u8; 52]) {
    let mut param = [0u8; 64];
    param[0] = 54; // digest_length = (512/144)*144/8
    param[2] = 1; // fanout
    param[3] = 1; // depth
    param[48..56].copy_from_slice(b"ZcashPoW");
    param[56..60].copy_from_slice(&144u32.to_le_bytes());
    param[60..64].copy_from_slice(&5u32.to_le_bytes());
    let mut h = IV;
    for i in 0..8 {
        h[i] ^= u64::from_le_bytes(param[8 * i..8 * i + 8].try_into().unwrap());
    }
    blake2b_compress(&mut h, &header[..128], 128, false);
    let mut rem = [0u8; 52];
    rem.copy_from_slice(&header[128..]);
    (h, rem)
}

// ---------------- solution post-processing (ports of solver_details) ----------------

fn duped(prf: &[u32; PROOFSIZE]) -> bool {
    let mut sorted = *prf;
    sorted.sort_unstable();
    sorted.windows(2).any(|w| w[1] <= w[0])
}

fn compress_solution(sol: &[u32; PROOFSIZE]) -> [u8; COMPRESSED_SOL_SIZE] {
    // 32 indices x 25 bits, big-endian bit packing (port of compress_solution)
    let mut out = [0u8; COMPRESSED_SOL_SIZE];
    let mut i = 0usize;
    let mut bits_left = 25u32;
    for b in out.iter_mut() {
        if bits_left >= 8 {
            bits_left -= 8;
            *b = (sol[i] >> bits_left) as u8;
        } else {
            let mut v = (sol[i] << (8 - bits_left)) as u8;
            i += 1;
            bits_left += 25 - 8;
            v |= (sol[i] >> bits_left) as u8;
            *b = v;
        }
    }
    out
}

// ---------------- wgpu plumbing ----------------

fn instance() -> &'static wgpu::Instance {
    static INSTANCE: OnceLock<wgpu::Instance> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY, // Vulkan / Metal / DX12
            ..Default::default()
        })
    })
}

fn adapters() -> Vec<wgpu::Adapter> {
    instance()
        .enumerate_adapters(wgpu::Backends::PRIMARY)
        .into_iter()
        .filter(|a| a.get_info().device_type != wgpu::DeviceType::Cpu)
        .collect()
}

thread_local! {
    static ADAPTER_IDX: Cell<u32> = const { Cell::new(0) };
}

pub struct EqSolver {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>, // digitH, digitR r=1..4, digitK
    bind_group: wgpu::BindGroup,
    params_buf: wgpu::Buffer,
    sols_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    workgroups: u32,
}

const SOLS_BYTES: u64 = 4 + (MAXSOLS * PROOFSIZE * 4) as u64;

fn create_solver(nthreads: u32) -> Result<EqSolver, String> {
    let nthreads = if nthreads == 0 { DEFAULT_NTHREADS } else { nthreads };
    let nthreads = nthreads.div_ceil(WORKGROUP) * WORKGROUP;

    let all = adapters();
    let idx = ADAPTER_IDX.with(|c| c.get()) as usize;
    let adapter = all.get(idx).ok_or_else(|| format!("no adapter {idx}"))?;

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("eqwgpu1445"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
        },
        None,
    ))
    .map_err(|e| e.to_string())?;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("solver"),
        source: wgpu::ShaderSource::Wgsl(include_str!("solver.wgsl").into()),
    });

    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage(0, true),
            storage(1, false),
            storage(2, false),
            storage(3, false),
            storage(4, false),
        ],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });

    let buf = |label: &str, size: u64, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    use wgpu::BufferUsages as U;
    let params_buf = buf("params", 29 * 4, U::STORAGE | U::COPY_DST);
    let heap0 = buf("heap0", HEAP_WORDS * 4, U::STORAGE);
    let heap1 = buf("heap1", HEAP_WORDS * 4, U::STORAGE);
    let nslots = buf("nslots", 2 * NBUCKETS * 4, U::STORAGE);
    let sols_buf = buf("sols", SOLS_BYTES, U::STORAGE | U::COPY_DST | U::COPY_SRC);
    let staging_buf = buf("staging", SOLS_BYTES, U::MAP_READ | U::COPY_DST);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: heap0.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: heap1.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: nslots.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: sols_buf.as_entire_binding() },
        ],
    });

    let mk_pipeline = |entry: &str, round: Option<u32>| {
        let mut constants = std::collections::HashMap::from([("NTHREADS".to_string(), nthreads as f64)]);
        if let Some(r) = round {
            constants.insert("ROUND".to_string(), r as f64);
        }
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(&layout),
            module: &module,
            entry_point: entry,
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        })
    };
    let mut pipelines = vec![mk_pipeline("digitH", None)];
    for r in 1..=4 {
        pipelines.push(mk_pipeline("digitR", Some(r)));
    }
    pipelines.push(mk_pipeline("digitK", None));

    Ok(EqSolver {
        device,
        queue,
        pipelines,
        bind_group,
        params_buf,
        sols_buf,
        staging_buf,
        workgroups: nthreads / WORKGROUP,
    })
}

fn solve(
    s: &EqSolver,
    header: &[u8],
    nonce: u32,
    mut on_solution: impl FnMut(&[u8; COMPRESSED_SOL_SIZE]) -> bool,
) -> i32 {
    if header.len() != HEADER_LEN {
        // ponytail: only the 180-byte algo-v1 header is supported (the only
        // live format); generalize the remainder handling if EXCC forks again
        return -1;
    }
    let mut hdr = [0u8; HEADER_LEN];
    hdr.copy_from_slice(header);
    hdr[NONCE_OFFSET..NONCE_OFFSET + 4].copy_from_slice(&nonce.to_le_bytes());

    let (h, rem) = equihash_midstate(&hdr);
    let mut params = [0u8; 29 * 4];
    for i in 0..8 {
        params[8 * i..8 * i + 8].copy_from_slice(&h[i].to_le_bytes());
    }
    params[64..116].copy_from_slice(&rem);
    s.queue.write_buffer(&s.params_buf, 0, &params);
    s.queue.write_buffer(&s.sols_buf, 0, &[0u8; 4]); // reset nsols

    let mut encoder = s.device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_bind_group(0, &s.bind_group, &[]);
        for p in &s.pipelines {
            pass.set_pipeline(p);
            pass.dispatch_workgroups(s.workgroups, 1, 1);
        }
    }
    encoder.copy_buffer_to_buffer(&s.sols_buf, 0, &s.staging_buf, 0, SOLS_BYTES);
    s.queue.submit([encoder.finish()]);

    let slice = s.staging_buf.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    s.device.poll(wgpu::Maintain::Wait);
    let found;
    {
        let data = slice.get_mapped_range();
        let nsols = u32::from_le_bytes(data[0..4].try_into().unwrap()).min(MAXSOLS as u32) as usize;
        let mut n = 0;
        'outer: for i in 0..nsols {
            let mut prf = [0u32; PROOFSIZE];
            for (j, w) in prf.iter_mut().enumerate() {
                let o = 4 + (i * PROOFSIZE + j) * 4;
                *w = u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
            }
            if duped(&prf) {
                continue;
            }
            n += 1;
            if on_solution(&compress_solution(&prf)) {
                break 'outer;
            }
        }
        found = n;
    }
    s.staging_buf.unmap();
    found
}

// ---------------- C ABI (matches eqcuda1445.h) ----------------

type SolutionCb = unsafe extern "C" fn(user_data: *mut c_void, solution: *mut c_void) -> i32;

/// # Safety
/// Called from C; returns an opaque solver handle or NULL.
#[no_mangle]
pub unsafe extern "C" fn eq_create(nthreads: u32) -> *mut EqSolver {
    match std::panic::catch_unwind(|| create_solver(nthreads)) {
        Ok(Ok(s)) => Box::into_raw(Box::new(s)),
        Ok(Err(e)) => {
            eprintln!("eqwgpu1445: {e}");
            std::ptr::null_mut()
        }
        Err(_) => {
            eprintln!("eqwgpu1445: panic in eq_create");
            std::ptr::null_mut()
        }
    }
}

/// # Safety
/// `solver` must come from eq_create.
#[no_mangle]
pub unsafe extern "C" fn eq_destroy(solver: *mut EqSolver) {
    if !solver.is_null() {
        drop(Box::from_raw(solver));
    }
}

/// # Safety
/// `solver` from eq_create; `header` points to `header_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn eq_solve(
    solver: *mut EqSolver,
    header: *const c_void,
    header_len: u32,
    nonce: u32,
    on_solution: Option<SolutionCb>,
    user_data: *mut c_void,
) -> i32 {
    if solver.is_null() || header.is_null() {
        return -1;
    }
    let s = &*solver;
    let hdr = std::slice::from_raw_parts(header as *const u8, header_len as usize);
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        solve(s, hdr, nonce, |csol| {
            if let Some(cb) = on_solution {
                cb(user_data, csol.as_ptr() as *mut c_void) != 0
            } else {
                false
            }
        })
    }));
    match res {
        Ok(n) => n,
        Err(_) => {
            eprintln!("eqwgpu1445: panic in eq_solve");
            -3
        }
    }
}

#[no_mangle]
pub extern "C" fn eq_adapter_count() -> u32 {
    adapters().len() as u32
}

/// # Safety
/// `buf` points to `len` writable bytes; writes a NUL-terminated name.
#[no_mangle]
pub unsafe extern "C" fn eq_adapter_name(index: u32, buf: *mut u8, len: u32) -> i32 {
    let all = adapters();
    let Some(a) = all.get(index as usize) else {
        return -1;
    };
    let info = a.get_info();
    let name = format!("{} ({:?})", info.name, info.backend);
    let bytes = name.as_bytes();
    let n = bytes.len().min(len as usize - 1);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, n);
    *buf.add(n) = 0;
    0
}

#[no_mangle]
pub extern "C" fn eq_set_adapter(index: u32) {
    ADAPTER_IDX.with(|c| c.set(index));
}

// ---------------- tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates_and_compiles_to_spirv() {
        let module = wgpu::naga::front::wgsl::parse_str(include_str!("solver.wgsl"))
            .expect("parse solver WGSL");
        let info = wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            Default::default(),
        )
        .validate(&module)
        .expect("validate solver WGSL");
        for (entry_point, round) in [
            ("digitH", None),
            ("digitR", Some(1)),
            ("digitR", Some(2)),
            ("digitR", Some(3)),
            ("digitR", Some(4)),
            ("digitK", None),
        ] {
            let mut constants = wgpu::naga::back::PipelineConstants::from([(
                "NTHREADS".to_string(),
                DEFAULT_NTHREADS as f64,
            )]);
            if let Some(round) = round {
                constants.insert("ROUND".to_string(), round as f64);
            }
            let (module, info) =
                wgpu::naga::back::pipeline_constants::process_overrides(&module, &info, &constants)
                    .unwrap_or_else(|e| panic!("resolve {entry_point} overrides: {e}"));
            let pipeline = wgpu::naga::back::spv::PipelineOptions {
                shader_stage: wgpu::naga::ShaderStage::Compute,
                entry_point: entry_point.to_string(),
            };
            wgpu::naga::back::spv::write_vec(&module, &info, &Default::default(), Some(&pipeline))
                .unwrap_or_else(|e| panic!("compile {entry_point} to SPIR-V: {e}"));
        }
    }

    // full hash of leaf index: blake2b(header || le32(idx/3)), take 18-byte
    // sub-hash idx%3 — used to verify Wagner conditions on GPU solutions
    fn leaf_hash(h: &[u64; 8], rem: &[u8; 52], idx: u32) -> [u8; 18] {
        let mut hh = *h;
        let mut block = [0u8; 128];
        block[..52].copy_from_slice(rem);
        block[52..56].copy_from_slice(&(idx / 3).to_le_bytes());
        blake2b_compress(&mut hh, &block, 184, true);
        let mut digest = [0u8; 56];
        for i in 0..7 {
            digest[8 * i..8 * i + 8].copy_from_slice(&hh[i].to_le_bytes());
        }
        let off = (idx % 3) as usize * 18;
        digest[off..off + 18].try_into().unwrap()
    }

    #[test]
    fn gpu_solutions_verify() {
        let mut header = [0x42u8; HEADER_LEN];
        let s = create_solver(0).expect("create solver");
        let mut total = 0;
        for nonce in 0u32..10 {
            header[NONCE_OFFSET..NONCE_OFFSET + 4].copy_from_slice(&nonce.to_le_bytes());
            let (h, rem) = equihash_midstate(&header);
            let n = solve(&s, &header, nonce, |csol| {
                // uncompress: 32 big-endian 25-bit indices
                let mut idxs = [0u32; PROOFSIZE];
                for (i, v) in idxs.iter_mut().enumerate() {
                    let bit = i * 25;
                    for b in 0..25 {
                        let bitpos = bit + b;
                        if csol[bitpos / 8] >> (7 - bitpos % 8) & 1 == 1 {
                            *v |= 1 << (24 - b);
                        }
                    }
                }
                // strictly ordered first elements & no dupes
                assert!(idxs.windows(2).all(|w| w[0] != w[1]));
                // xor of all 32 leaf hashes must be zero in every bit
                let mut acc = [0u8; 18];
                for &idx in &idxs {
                    let lh = leaf_hash(&h, &rem, idx);
                    for (a, b) in acc.iter_mut().zip(lh.iter()) {
                        *a ^= b;
                    }
                }
                assert_eq!(acc, [0u8; 18], "solution xor must be zero");
                false
            });
            assert!(n >= 0, "solve failed: {n}");
            total += n;
        }
        // ~2 solutions per nonce expected; 10 nonces with none would be
        // astronomically unlikely
        assert!(total > 0, "no solutions found in 10 nonces");
        println!("verified {total} solutions over 10 nonces");
    }
}
