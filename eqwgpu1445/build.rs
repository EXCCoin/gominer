use std::env;
use std::fs;
use std::path::PathBuf;

fn replace_exact(source: String, from: &str, to: &str, expected: usize) -> String {
    let found = source.matches(from).count();
    assert_eq!(
        found, expected,
        "expected {expected} occurrences of {from:?}, found {found}"
    );
    source.replace(from, to)
}

struct CompactVariant {
    output: &'static str,
    heads: u32,
    capacity: u32,
    staged_words: u32,
    workgroup: u32,
    small_part_bits: u32,
    big_part_bits: u32,
    small_capacity: u32,
    big_capacity: u32,
    head_bits: u32,
}

fn derive_compact(source: &str, variant: &CompactVariant) -> String {
    let mut output = source.to_owned();
    output = replace_exact(
        output,
        "array<atomic<u32>, 1024>",
        &format!("array<atomic<u32>, {}>", variant.heads),
        1,
    );
    output = replace_exact(
        output,
        "array<u32, 2816>",
        &format!("array<u32, {}>", variant.capacity),
        1,
    );
    output = replace_exact(
        output,
        "array<u32, 11264>",
        &format!("array<u32, {}>", variant.staged_words),
        1,
    );
    output = replace_exact(
        output,
        "@workgroup_size(1024)",
        &format!("@workgroup_size({})", variant.workgroup),
        2,
    );
    output = replace_exact(
        output,
        "select(1u, 2u, big_round)",
        &format!(
            "select({}u, {}u, big_round)",
            variant.small_part_bits, variant.big_part_bits
        ),
        1,
    );
    output = replace_exact(
        output,
        "select(2816u, 2240u, big_round)",
        &format!(
            "select({}u, {}u, big_round)",
            variant.small_capacity, variant.big_capacity
        ),
        1,
    );
    output = replace_exact(
        output,
        "slot += 1024u",
        &format!("slot += {}u", variant.workgroup),
        2,
    );
    output = replace_exact(
        output,
        "pos1 += 1024u",
        &format!("pos1 += {}u", variant.workgroup),
        2,
    );
    output = replace_exact(
        output,
        "key >> 10u",
        &format!("key >> {}u", variant.head_bits),
        1,
    );
    output = replace_exact(
        output,
        "key & 1023u",
        &format!("key & {}u", variant.heads - 1),
        1,
    );
    replace_exact(
        output,
        "head += 1024u",
        &format!("head += {}u", variant.workgroup),
        1,
    )
}

fn main() {
    const INPUT: &str = "src/big_solver.wgsl";
    println!("cargo:rerun-if-changed={INPUT}");

    // Derive two compact versions of the large-bucket kernel. More bucket
    // partitions let each round use fewer hash heads without changing the
    // collision key. digitK has its own small table and keeps 1024 heads.
    let source = fs::read_to_string(INPUT).expect("read large-bucket WGSL");
    let variants = [
        CompactVariant {
            output: "big_solver_compact32.wgsl",
            heads: 512,
            capacity: 1408,
            staged_words: 5632,
            workgroup: 512,
            small_part_bits: 2,
            big_part_bits: 3,
            small_capacity: 1408,
            big_capacity: 1120,
            head_bits: 9,
        },
        CompactVariant {
            output: "big_solver_compact16.wgsl",
            heads: 256,
            capacity: 767,
            staged_words: 3068,
            workgroup: 256,
            small_part_bits: 3,
            big_part_bits: 4,
            small_capacity: 767,
            big_capacity: 610,
            head_bits: 8,
        },
    ];
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    for variant in variants {
        let compact = derive_compact(&source, &variant);
        fs::write(out_dir.join(variant.output), compact)
            .unwrap_or_else(|e| panic!("write {}: {e}", variant.output));
    }
}
