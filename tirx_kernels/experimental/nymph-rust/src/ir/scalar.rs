//! Scalar expression IR: `Var`, `ScalarExpr`, `ScalarValue`, `ScalarInitial`.
//!
//! KEY DESIGN POINT — identity. In Python `Var` is `eq=False` (two vars with the
//! same fields are still DIFFERENT — identity matters: a var is keyed by object
//! identity in the scalar environment). Rust has no object identity, so we give
//! each `Var` a stable `VarId` and make equality/hash use ONLY that id. The Var
//! also carries its own data (binding, dtype) inline — it's tiny and `Copy`, so we
//! pass it around by value; "refer to the same var" = a `Var` value with the same
//! id. (The builder is the source of fresh ids.)

use super::dtype::{ScalarDType, ScalarOp, ScopeValueKind, VarBinding};
use super::tensor::TensorSlice;
use std::hash::{Hash, Hasher};

/// Stable identity of a `Var` (assigned by the builder from a counter).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VarId(pub u32);

/// `Var` — a scalar variable (loop / scalar / task). Carries its data inline; its
/// identity is `id` only (see the module note).
#[derive(Clone, Copy, Debug)]
pub struct Var {
    pub id: VarId,
    pub binding: VarBinding,
    pub dtype: ScalarDType,
}

// Equality/hash by id ONLY — this is the whole point of the identity model.
impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Var {}
impl Hash for Var {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

/// `ScalarExpr` — an operation over scalar values (e.g. `task * 16 + k`).
/// Recursive: its args are `ScalarValue`s, which may themselves be exprs.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ScalarExpr {
    pub op: ScalarOp,
    pub args: Vec<ScalarValue>,
}

/// Python floor division (rounds toward -inf).
fn floor_div(a: i64, b: i64) -> i64 {
    let q = a / b;
    let r = a % b;
    if r != 0 && ((r < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}
/// Python modulo (sign follows divisor).
fn floor_mod(a: i64, b: i64) -> i64 {
    let m = a % b;
    if m != 0 && ((m < 0) != (b < 0)) {
        m + b
    } else {
        m
    }
}

/// Pure `ScalarOp` application on i64 operands: floor div/mod (Python
/// semantics, sign follows divisor), bitwise And/Or/Xor, logical Not,
/// comparison results as 0/1. The single source of the scalar ALU — shared by
/// the interpreter's evaluator and the static thread filter. Division or
/// modulo by zero is the only failure.
pub fn apply_scalar_op(op: ScalarOp, args: &[i64]) -> Result<i64, &'static str> {
    Ok(match op {
        ScalarOp::Add => args[0].wrapping_add(args[1]),
        ScalarOp::Sub => args[0].wrapping_sub(args[1]),
        ScalarOp::Mul => args[0].wrapping_mul(args[1]),
        ScalarOp::Xor => args[0] ^ args[1],
        ScalarOp::And => args[0] & args[1],
        ScalarOp::Or => args[0] | args[1],
        ScalarOp::Eq => (args[0] == args[1]) as i64,
        ScalarOp::Ne => (args[0] != args[1]) as i64,
        ScalarOp::Lt => (args[0] < args[1]) as i64,
        ScalarOp::Le => (args[0] <= args[1]) as i64,
        ScalarOp::Gt => (args[0] > args[1]) as i64,
        ScalarOp::Ge => (args[0] >= args[1]) as i64,
        ScalarOp::FloorDiv => {
            if args[1] == 0 {
                return Err("division by zero");
            }
            floor_div(args[0], args[1])
        }
        ScalarOp::Mod => {
            if args[1] == 0 {
                return Err("modulo by zero");
            }
            floor_mod(args[0], args[1])
        }
        ScalarOp::Neg => -args[0],
        ScalarOp::Not => (args[0] == 0) as i64,
        ScalarOp::Select => {
            if args[0] != 0 {
                args[1]
            } else {
                args[2]
            }
        }
        ScalarOp::Min => args[0].min(args[1]),
        ScalarOp::Max => args[0].max(args[1]),
    })
}

/// `ScalarValue = int | Var | ScalarExpr | ScopeValue` — the Python union type,
/// expressed as a Rust enum (Python dispatched on isinstance; Rust matches).
/// `Expr` is boxed because the type is recursive (an expr contains scalar values).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ScalarValue {
    Int(i64),
    Var(Var),
    Expr(Box<ScalarExpr>),
    Scope(ScopeValueKind),
}

impl ScalarValue {
    /// Convenience: wrap an expr (boxes it).
    pub fn expr(op: ScalarOp, args: Vec<ScalarValue>) -> ScalarValue {
        ScalarValue::Expr(Box::new(ScalarExpr { op, args }))
    }
}

// Let plain ints be used wherever a ScalarValue is expected: `ScalarValue::from(5)`.
impl From<i64> for ScalarValue {
    fn from(v: i64) -> Self {
        ScalarValue::Int(v)
    }
}
impl From<Var> for ScalarValue {
    fn from(v: Var) -> Self {
        ScalarValue::Var(v)
    }
}

/// `ScalarInitial = ScalarValue | TensorSlice` — the initial value of a scalar_def
/// (either a scalar, or a 1-element GMEM tensor slice to load from).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ScalarInitial {
    Value(ScalarValue),
    Tensor(TensorSlice),
}
