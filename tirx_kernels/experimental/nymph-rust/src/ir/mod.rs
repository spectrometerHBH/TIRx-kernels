//! The nymph IR — a faithful Rust port of `ir/ir.py`.
//!
//! Layout (mirrors the Python module structure):
//! - `dtype`  — the simple enums (MemorySpace, DType, ScalarOp, ...)
//! - `scalar` — Var (identity model), ScalarExpr, ScalarValue, ScalarInitial
//! - `tensor` — Tensor (table/id model), TensorSlice, Layout, TmemOperand
//! - `mbar`   — MBar (table/id model), MBarRef
//! - `stmt`   — the big Stmt enum (~41 variants)
//! - `kernel` — Kernel (owns the tensor/mbar tables)

pub mod codegen;
pub mod dtype;
pub mod kernel;
pub mod mbar;
pub mod scalar;
pub mod scheduler;
pub mod stmt;
pub mod tensor;
pub mod validate;

// Re-export everything so callers can write `nymph::ir::Tensor` etc.
pub use codegen::*;
pub use dtype::*;
pub use kernel::*;
pub use mbar::*;
pub use scalar::*;
pub use scheduler::*;
pub use stmt::*;
pub use tensor::*;
pub use validate::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Build a tiny IR by hand to prove all the types compose and the identity
    /// model works. (Once the builder exists, this is what it produces.)
    #[test]
    fn assemble_a_tiny_kernel() {
        // A GMEM input tensor C (f32, 256x256) and an SMEM operand A (f16, 256x16),
        // each heap-allocated once and shared via Arc.
        let c = Arc::new(Tensor {
            id: 0,
            space: MemorySpace::Gmem,
            dtype: DType::F32,
            shape: vec![256, 256],
            layout: None,
            byte_offset: None,
            reg_frag: None,
        });
        let a = Arc::new(Tensor {
            id: 1,
            space: MemorySpace::Smem,
            dtype: DType::F16,
            shape: vec![256, 16],
            layout: None,
            byte_offset: Some(0),
            reg_frag: None,
        });

        // A loop variable `k` (identity is its id).
        let k = Var {
            id: VarId(0),
            binding: VarBinding::Loop,
            dtype: ScalarDType::I32,
        };
        let k_again = k; // a copy refers to the SAME var (equal by id)
        assert_eq!(k, k_again);
        let other = Var {
            id: VarId(1),
            binding: VarBinding::Loop,
            dtype: ScalarDType::I32,
        };
        assert_ne!(k, other); // different id => different var

        // A symbolic offset: k * 16  (a ScalarExpr inside a ScalarValue).
        let offset = ScalarValue::expr(
            ScalarOp::Mul,
            vec![ScalarValue::Var(k), ScalarValue::Int(16)],
        );

        // A slice over A — note we can read the tensor's data right here.
        let a_slice = TensorSlice {
            tensor: Arc::clone(&a),
            offsets: vec![ScalarValue::Int(0), offset.clone()],
            shape: vec![ScalarValue::Int(256), ScalarValue::Int(16)],
        };
        // Cross-reference works at construction (this is the whole point of Arc):
        assert_eq!(a_slice.tensor.space, MemorySpace::Smem);
        assert_eq!(a_slice.tensor.dtype, DType::F16);
        // Two Arc clones of the same tensor are "the same tensor" (equal by id).
        let a2 = Arc::clone(&a);
        assert_eq!(*a_slice.tensor, *a2);

        let body = vec![
            Stmt::ForLoop {
                no_unroll: false,
                var: k,
                start: ScalarValue::Int(0),
                stop: ScalarValue::Int(16),
                step: ScalarValue::Int(1),
                body: vec![Stmt::RegStore {
                    dst: a_slice.clone(),
                    src: a_slice.clone(),
                }],
                unroll: false,
            },
            Stmt::CtaSync,
        ];

        let kernel = Kernel {
            name: "tiny".to_string(),
            args: vec![Arc::clone(&c)],
            body,
            num_warps: 12,
            smem_size_bytes: 256 * 16 * 2,
            launch_shape: vec![2],
            cluster_shape: vec![2],
            smem_pool: false,
        };

        // Arg tensor data is reachable directly through its Arc.
        assert_eq!(kernel.args[0].dtype, DType::F32);
        assert_eq!(kernel.launch_cta_count(), 2);
        // The control node exposes its child body for generic walks.
        assert_eq!(kernel.body[0].child_bodies().len(), 1);
        assert_eq!(kernel.body[1].child_bodies().len(), 0);
    }

    fn kernel(body: Vec<Stmt>, num_warps: u32) -> Kernel {
        Kernel {
            name: "t".to_string(),
            args: vec![],
            body,
            num_warps,
            smem_size_bytes: 0,
            launch_shape: vec![2],
            cluster_shape: vec![2],
            smem_pool: false,
        }
    }

    #[test]
    fn valid_kernel_passes() {
        let s = Var {
            id: VarId(0),
            binding: VarBinding::Scalar,
            dtype: ScalarDType::I32,
        };
        let body = vec![
            Stmt::ScalarDef {
                var: s,
                initial: ScalarInitial::Value(ScalarValue::Int(0)),
            },
            Stmt::ScalarStore {
                var: s,
                value: ScalarValue::Int(5),
            },
            Stmt::CtaSync,
        ];
        assert!(kernel(body, 4).validate().is_ok());
    }

    #[test]
    fn rejects_bad_num_warps() {
        // 6 is not a multiple of 4 -> local check fails.
        let e = kernel(vec![], 6).validate().unwrap_err();
        assert!(e.message.contains("num_warps"), "{}", e.message);
    }

    #[test]
    fn rejects_undefined_var() {
        // Storing to a var that was never defined -> var-defs walk fails.
        let s = Var {
            id: VarId(7),
            binding: VarBinding::Scalar,
            dtype: ScalarDType::I32,
        };
        let e = kernel(
            vec![Stmt::ScalarStore {
                var: s,
                value: ScalarValue::Int(1),
            }],
            4,
        )
        .validate()
        .unwrap_err();
        assert!(e.message.contains("defined before use"), "{}", e.message);
    }

    #[test]
    fn let_binding_validates_and_feeds_later_defs() {
        // A `ScalarLet` defines its var (later uses resolve) and stays single-shot.
        let s = Var {
            id: VarId(0),
            binding: VarBinding::Scalar,
            dtype: ScalarDType::I32,
        };
        let t = Var {
            id: VarId(1),
            binding: VarBinding::Scalar,
            dtype: ScalarDType::I32,
        };
        let body = vec![
            Stmt::ScalarLet {
                var: s,
                value: ScalarValue::expr(
                    ScalarOp::Add,
                    vec![ScalarValue::Int(1), ScalarValue::Int(2)],
                ),
            },
            Stmt::ScalarDef {
                var: t,
                initial: ScalarInitial::Value(ScalarValue::Var(s)),
            },
            Stmt::CtaSync,
        ];
        assert!(kernel(body, 4).validate().is_ok());
    }

    #[test]
    fn rejects_scalar_store_to_let_var() {
        // The let contract is single-assignment: a ScalarStore to the var is
        // rejected even though the var is defined at that point.
        let s = Var {
            id: VarId(0),
            binding: VarBinding::Scalar,
            dtype: ScalarDType::I32,
        };
        let e = kernel(
            vec![
                Stmt::ScalarLet {
                    var: s,
                    value: ScalarValue::Int(0),
                },
                Stmt::ScalarStore {
                    var: s,
                    value: ScalarValue::Int(5),
                },
            ],
            4,
        )
        .validate()
        .unwrap_err();
        assert!(e.message.contains("single assignment"), "{}", e.message);
    }

    #[test]
    fn rejects_cta_sync_in_warp_scope() {
        // cta_sync inside a warp-scope init block -> scope walk fails.
        let body = vec![Stmt::KernelInit {
            body: vec![Stmt::CtaSync],
            warp: Some(0),
            lane: None,
            elected: false,
        }];
        let e = kernel(body, 4).validate().unwrap_err();
        assert!(
            e.message.contains("cta_sync must be in CTA scope"),
            "{}",
            e.message
        );
    }

    #[test]
    fn rejects_cta_sync_inside_role() {
        let body = vec![Stmt::Role {
            else_body: Vec::new(),
            body: vec![Stmt::CtaSync],
            warp: None,
            warpgroup: None,
            elected: false,
            maxnreg: None,
        }];
        let e = kernel(body, 4).validate().unwrap_err();
        assert!(
            e.message.contains("cta_sync cannot be used inside role"),
            "{}",
            e.message
        );
    }
}

/// Mechanical anti-drift gate: every `Stmt` variant must be handled (or
/// explicitly rejected) in each consumer — validate.rs, codegen.rs, the
/// interpreter dispatch (registry.rs `stmt_kind`), and the protocol checker's
/// metadata walk (checker.rs `walk_tensors`). The variant list is PARSED out
/// of `stmt.rs`, so adding a variant without touching the consumers fails
/// this test. (Three of the four matches are additionally compiler-enforced
/// exhaustive — no wildcard arms — so a missing arm is a compile error; this
/// test guards the remaining text-level drift and documents the contract.)
#[cfg(test)]
mod variant_coverage_tests {
    use std::collections::BTreeSet;

    /// Parse the `Stmt` variant names out of stmt.rs: the enum body at 4-space
    /// indent, `Name {` (struct) or `Name,` (unit), doc/comment lines skipped.
    fn stmt_variants() -> BTreeSet<String> {
        let src = include_str!("stmt.rs");
        let start = src.find("pub enum Stmt {").expect("Stmt enum");
        let mut out = BTreeSet::new();
        for line in src[start..].lines().skip(1) {
            if line.trim_end() == "}" {
                break;
            }
            let t = line.trim_start();
            if t.is_empty() || t.starts_with("//") || t.starts_with('#') {
                continue;
            }
            // Variant decls are at exactly 4-space indent (fields are deeper).
            if !line.starts_with("    ") || line.starts_with("     ") {
                continue;
            }
            let name: String = t
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if name.chars().next().is_some_and(char::is_uppercase) {
                out.insert(name);
            }
        }
        out
    }

    #[test]
    fn every_stmt_variant_is_consumed_everywhere() {
        let consumers: [(&str, &str); 4] = [
            ("validate.rs", include_str!("validate.rs")),
            ("codegen.rs", include_str!("codegen.rs")),
            (
                "registry.rs (interpreter dispatch)",
                include_str!("../interpreter/registry.rs"),
            ),
            (
                "checker.rs (protocol checker)",
                include_str!("../interpreter/checker.rs"),
            ),
        ];
        let variants = stmt_variants();
        assert!(
            variants.len() >= 60,
            "expected the full Stmt variant set, parsed {}",
            variants.len()
        );
        for (label, src) in consumers {
            for v in &variants {
                assert!(
                    src.contains(v.as_str()),
                    "Stmt::{v} is not handled (or explicitly rejected) in {label}"
                );
            }
        }
    }
}
