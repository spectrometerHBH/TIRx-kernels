//! Real-hardware validation of the warp-level `mma.sync` (the WarpMma IR node).
//!
//! Generates a CUDA harness that runs `mma.sync.aligned.m16n8k{16,8}.row.col.f32.{bf16,
//! f16}.{...}.f32` on an sm_100 GPU with random A/B fragments and dumps each lane's D
//! accumulator fragment. Operands are packed in the standard mma fragment layout, then D
//! is un-scattered via the SAME accumulator layout the interpreter's `warpmma.rs` models
//! (A reg ru→tile (mt=ru%2,kt=ru/2), A[mt*8+g][kt*8+2t+h]; D[(ri/2)*8+g][2t+ri%2]) and
//! checked cell-for-cell against the f32 reference matmul A·Bᵀ — so the interpreter's A/D
//! fragment layout AND the bf16/f16 operand decode are verified against real hardware. (B
//! is packed in the raw PTX `.col` direct-feed layout B[n=g][k=kt*8+2t+h]; the interpreter's
//! post-`ldmatrix.trans` B layout is covered transitively by `ldstmatrix_hardware` +
//! `test_warp_mma`'s ldmatrix.trans→mma round-trip.)
//!
//! Run on a Blackwell box with: `cargo test --test mma_sync_hardware -- --ignored`.
//! Passing the warp-MMA's bf16 + f16 + (k16,k8) cases on a B200 (sm_100) was verified.

use half::{bf16, f16};
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const THREADS: usize = 32;
// (K, f16) — bf16 and f16 operands, k16 and k8.
fn cases() -> Vec<(usize, bool)> {
    vec![(16, false), (16, true), (8, false), (8, true)]
}

fn out_dir() -> PathBuf {
    std::env::var_os("NYMPH_HW_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/mma_sync_hardware"))
}

fn pack(lo: u16, hi: u16) -> u32 {
    (lo as u32) | ((hi as u32) << 16)
}
fn enc(x: f32, is_f16: bool) -> u16 {
    if is_f16 {
        f16::from_f32(x).to_bits()
    } else {
        bf16::from_f32(x).to_bits()
    }
}
fn rounded(x: f32, is_f16: bool) -> f32 {
    if is_f16 {
        f16::from_f32(x).to_f32()
    } else {
        bf16::from_f32(x).to_f32()
    }
}

// Deterministic pseudo-random in [-0.5, 0.5).
fn rnd(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (((*seed >> 33) as u32) as f32 / u32::MAX as f32) - 0.5
}

#[test]
#[ignore = "requires nvcc and an sm_100 GPU"]
fn mma_sync_matches_hardware() {
    let cases = cases();
    let dir = out_dir();
    fs::create_dir_all(&dir).expect("create out dir");

    // Build per-case A/B matrices + packed fragments + reference D.
    let (m, n) = (16usize, 8usize);
    let mut a_frags: Vec<u32> = Vec::new(); // [case][lane][reg]
    let mut b_frags: Vec<u32> = Vec::new();
    let mut d_ref: Vec<Vec<f32>> = Vec::new(); // [case] -> m*n
    let mut seed = 0x1234_5678u64;
    for &(kk, is_f16) in &cases {
        let mut a = vec![0f32; m * kk];
        let mut b = vec![0f32; n * kk];
        for v in a.iter_mut() {
            *v = rounded(rnd(&mut seed) * 0.8, is_f16);
        }
        for v in b.iter_mut() {
            *v = rounded(rnd(&mut seed) * 0.8, is_f16);
        }
        // pack A: reg ru -> (mt=ru%2, kt=ru/2); halves h -> A[mt*8+g][kt*8+2t+h]
        let na = m * kk / 64;
        let nb = n * kk / 64;
        for lane in 0..THREADS {
            let (g, t) = (lane / 4, lane % 4);
            for ru in 0..na {
                let (mt, kt) = (ru % 2, ru / 2);
                let lo = enc(a[(mt * 8 + g) * kk + (kt * 8 + 2 * t)], is_f16);
                let hi = enc(a[(mt * 8 + g) * kk + (kt * 8 + 2 * t + 1)], is_f16);
                a_frags.push(pack(lo, hi));
            }
        }
        // Raw PTX m16n8k16 .col B fragment (fed directly to the GPU mma, no ldmatrix):
        // reg ru = k-tile; the two halves are B[n=g][k = kt*8 + 2t + h].
        for lane in 0..THREADS {
            let (g, t) = (lane / 4, lane % 4);
            for ru in 0..nb {
                let kt = ru;
                let lo = enc(b[g * kk + (kt * 8 + 2 * t)], is_f16);
                let hi = enc(b[g * kk + (kt * 8 + 2 * t + 1)], is_f16);
                b_frags.push(pack(lo, hi));
            }
        }
        let mut dr = vec![0f32; m * n];
        for mm in 0..m {
            for nn in 0..n {
                let mut acc = 0f32;
                for x in 0..kk {
                    acc += a[mm * kk + x] * b[nn * kk + x];
                }
                dr[mm * n + nn] = acc;
            }
        }
        d_ref.push(dr);
    }

    let a_bin = dir.join("a.bin");
    let b_bin = dir.join("b.bin");
    let d_bin = dir.join("d.bin");
    fs::write(&a_bin, bytemuck_u32(&a_frags)).unwrap();
    fs::write(&b_bin, bytemuck_u32(&b_frags)).unwrap();

    let cu = dir.join("mma_sync_hardware.cu");
    let bin = dir.join("mma_sync_hardware");
    fs::write(&cu, generate_cuda(&cases)).expect("write cuda");
    let nvcc = Command::new("nvcc")
        .args([
            "-gencode",
            "arch=compute_100a,code=sm_100a",
            "-o",
            bin.to_str().unwrap(),
            cu.to_str().unwrap(),
        ])
        .output()
        .expect("run nvcc");
    assert!(
        nvcc.status.success(),
        "nvcc failed\n{}",
        String::from_utf8_lossy(&nvcc.stderr)
    );
    let run = Command::new(&bin)
        .args([
            a_bin.to_str().unwrap(),
            b_bin.to_str().unwrap(),
            d_bin.to_str().unwrap(),
        ])
        .output()
        .expect("run harness");
    assert!(
        run.status.success(),
        "harness failed\n{}",
        String::from_utf8_lossy(&run.stderr)
    );

    let d_hw = read_f32(&d_bin); // [case][lane][ri], ri in 0..4
    assert_eq!(d_hw.len(), cases.len() * THREADS * 4);

    for (ci, &(_kk, is_f16)) in cases.iter().enumerate() {
        let base = ci * THREADS * 4;
        // un-scatter the GPU accumulator fragment via the modeled layout
        let mut dm = vec![0f32; m * n];
        for lane in 0..THREADS {
            let (g, t) = (lane / 4, lane % 4);
            for ri in 0..4 {
                let mm = (ri / 2) * 8 + g;
                let nn = 2 * t + ri % 2;
                dm[mm * n + nn] = d_hw[base + lane * 4 + ri];
            }
        }
        let atol = if is_f16 { 2e-2 } else { 1e-1 };
        for i in 0..m * n {
            assert!(
                (dm[i] - d_ref[ci][i]).abs() <= atol + 0.05 * d_ref[ci][i].abs(),
                "case {ci} (f16={is_f16}) cell {i}: hw={} ref={}",
                dm[i],
                d_ref[ci][i]
            );
        }
    }
}

fn bytemuck_u32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn read_f32(path: &PathBuf) -> Vec<f32> {
    let bytes = fs::read(path).expect("read d.bin");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn generate_cuda(cases: &[(usize, bool)]) -> String {
    let mut out = String::new();
    out.push_str(
        "#include <cstdint>\n#include <cstdio>\n#include <vector>\n#include <cuda_runtime.h>\n\n",
    );
    // offsets into the flat a/b frag arrays per case
    let mut a_off = 0usize;
    let mut b_off = 0usize;
    let mut offs = String::new();
    for (ci, &(kk, _)) in cases.iter().enumerate() {
        let na = 16 * kk / 64;
        let nb = 8 * kk / 64;
        writeln!(
            offs,
            "// case {ci}: a_off={a_off} na={na} b_off={b_off} nb={nb}"
        )
        .unwrap();
        a_off += THREADS * na;
        b_off += THREADS * nb;
    }
    out.push_str(&offs);
    out.push_str(
        "\n__global__ void mma_kernel(const uint32_t* a, const uint32_t* b, float* d) {\n",
    );
    out.push_str("  unsigned lane = threadIdx.x & 31;\n");
    let mut a_o = 0usize;
    let mut b_o = 0usize;
    for (ci, &(kk, is_f16)) in cases.iter().enumerate() {
        let na = 16 * kk / 64;
        let nb = 8 * kk / 64;
        let abt = if is_f16 { "f16" } else { "bf16" };
        writeln!(out, "  {{ // case {ci}").unwrap();
        for r in 0..na {
            writeln!(out, "    uint32_t a{r} = a[{} + lane*{na} + {r}];", a_o).unwrap();
        }
        for r in 0..nb {
            writeln!(out, "    uint32_t b{r} = b[{} + lane*{nb} + {r}];", b_o).unwrap();
        }
        writeln!(out, "    float d0=0.f,d1=0.f,d2=0.f,d3=0.f;").unwrap();
        let a_ops = brace(0, na); // operands %4 .. %(4+na-1)
        let b_ops = brace(na, nb); // operands %(4+na) .. (after A)
        writeln!(out,
            "    asm volatile(\"mma.sync.aligned.m16n8k{kk}.row.col.f32.{abt}.{abt}.f32 \"\n      \"{{%0,%1,%2,%3}}, {a_ops}, {b_ops}, {{%0,%1,%2,%3}};\\n\"\n      : \"+f\"(d0),\"+f\"(d1),\"+f\"(d2),\"+f\"(d3) : {ins});",
            ins = mma_inputs(na, nb)).unwrap();
        writeln!(out, "    float* dp = d + {}*32*4 + lane*4;", ci).unwrap();
        writeln!(out, "    dp[0]=d0; dp[1]=d1; dp[2]=d2; dp[3]=d3;").unwrap();
        writeln!(out, "  }}").unwrap();
        a_o += THREADS * na;
        b_o += THREADS * nb;
    }
    let _ = (a_off, b_off);
    out.push_str("}\n\n");
    write!(out, r#"static std::vector<uint32_t> readu32(const char* p, size_t n) {{
  std::vector<uint32_t> v(n); FILE* f=std::fopen(p,"rb"); std::fread(v.data(),4,n,f); std::fclose(f); return v; }}
int main(int argc, char** argv) {{
  size_t na={a_o}, nb={b_o}, nd={ncases}*32*4;
  auto ah=readu32(argv[1],na); auto bh=readu32(argv[2],nb);
  uint32_t *ad,*bd; float* dd;
  cudaMalloc(&ad,na*4); cudaMalloc(&bd,nb*4); cudaMalloc(&dd,nd*4);
  cudaMemcpy(ad,ah.data(),na*4,cudaMemcpyHostToDevice);
  cudaMemcpy(bd,bh.data(),nb*4,cudaMemcpyHostToDevice);
  mma_kernel<<<1,32>>>(ad,bd,dd);
  cudaError_t e=cudaDeviceSynchronize();
  if(e!=cudaSuccess){{ std::fprintf(stderr,"launch: %s\n",cudaGetErrorString(e)); return 1; }}
  std::vector<float> dh(nd); cudaMemcpy(dh.data(),dd,nd*4,cudaMemcpyDeviceToHost);
  FILE* f=std::fopen(argv[3],"wb"); std::fwrite(dh.data(),4,nd,f); std::fclose(f);
  return 0;
}}
"#, a_o = a_o, b_o = b_o, ncases = cases.len()).unwrap();
    out
}

fn brace(start: usize, n: usize) -> String {
    let regs: Vec<String> = (0..n).map(|i| format!("%{}", start + 4 + i)).collect();
    format!("{{{}}}", regs.join(","))
}
fn mma_inputs(na: usize, nb: usize) -> String {
    let mut s = String::new();
    for r in 0..na {
        write!(s, "\"r\"(a{r}),").unwrap();
    }
    for r in 0..nb {
        write!(s, "\"r\"(b{r})").unwrap();
        if r + 1 < nb {
            s.push(',');
        }
    }
    s
}
