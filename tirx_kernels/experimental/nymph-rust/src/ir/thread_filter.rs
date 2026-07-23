//! Static per-thread evaluation of `If` predicates.
//!
//! Under the per-warp execution model, all thread dispatch is `If` over a
//! free-form scalar predicate. Validation rules that used to hang off `Role`
//! structure (wg_sync scope, tmem alloc warp scope, setmaxnreg warpgroup
//! coverage, ...) instead ask: *which threads of a CTA can enter this branch?*
//!
//! `static_thread_filter(cond, num_warps)` answers that by evaluating the
//! predicate at every (warp_id, lane_id) point of one CTA. Predicates built
//! purely from warp/lane-coordinate scope values and integer constants
//! resolve to `Known(ThreadSet)`. Anything touching a runtime value (`Var`,
//! tensor-loaded scalar) or a CTA-coordinate leaf (`CtaId`,
//! `CtaidInCluster`, `NvshmemMyPe`) makes the affected threads
//! indeterminate and the whole filter `Unknown` — callers treat Unknown
//! conservatively: static rules skip, runtime semantics own the check.
//!
//! One refinement keeps mixed predicates useful: absorption. `x & 0 == 0`
//! and `Select` with a known condition resolve without evaluating the
//! unknown side, so `warpgroup_id()==1 && <runtime cond>` still yields
//! Known-false for every thread outside warpgroup 1 — only if some thread
//! remains indeterminate does the filter degrade to Unknown.

use super::dtype::{ScalarOp, ScopeValueKind};
use super::scalar::{apply_scalar_op, ScalarValue};

/// A set of threads within one CTA, bit index = `warp_id * 32 + lane_id`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ThreadSet {
    num_warps: u32,
    bits: Vec<bool>,
}

impl ThreadSet {
    pub fn full(num_warps: u32) -> ThreadSet {
        ThreadSet {
            num_warps,
            bits: vec![true; (num_warps * 32) as usize],
        }
    }

    pub fn num_warps(&self) -> u32 {
        self.num_warps
    }

    pub fn contains(&self, warp: u32, lane: u32) -> bool {
        self.bits[(warp * 32 + lane) as usize]
    }

    pub fn count(&self) -> usize {
        self.bits.iter().filter(|&&b| b).count()
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    pub fn is_full_cta(&self) -> bool {
        self.bits.iter().all(|&b| b)
    }

    /// Set intersection; both sides must describe the same CTA geometry.
    pub fn intersect(&self, other: &ThreadSet) -> ThreadSet {
        assert_eq!(self.num_warps, other.num_warps);
        ThreadSet {
            num_warps: self.num_warps,
            bits: self
                .bits
                .iter()
                .zip(other.bits.iter())
                .map(|(&a, &b)| a && b)
                .collect(),
        }
    }

    /// Warps with at least one selected lane.
    pub fn warps_touched(&self) -> Vec<u32> {
        (0..self.num_warps)
            .filter(|&w| (0..32).any(|l| self.contains(w, l)))
            .collect()
    }

    pub fn is_full_warp(&self, warp: u32) -> bool {
        (0..32).all(|l| self.contains(warp, l))
    }

    /// Warps with all 32 lanes selected.
    pub fn full_warps(&self) -> Vec<u32> {
        (0..self.num_warps)
            .filter(|&w| self.is_full_warp(w))
            .collect()
    }

    /// Exactly one warp, all 32 lanes, nothing else.
    pub fn is_exactly_one_full_warp(&self) -> Option<u32> {
        let touched = self.warps_touched();
        match touched.as_slice() {
            [w] if self.is_full_warp(*w) => Some(*w),
            _ => None,
        }
    }

    /// Exactly one warpgroup (4 consecutive warps at a 4-warp boundary), all
    /// 128 lanes, nothing else.
    pub fn is_exactly_one_full_warpgroup(&self) -> Option<u32> {
        let touched = self.warps_touched();
        let first = *touched.first()?;
        let g = first / 4;
        let expect: Vec<u32> = (g * 4..g * 4 + 4).collect();
        if touched == expect && expect.iter().all(|&w| self.is_full_warp(w)) {
            Some(g)
        } else {
            None
        }
    }

    /// Non-empty and every touched warpgroup is fully covered (all 4 warps,
    /// all lanes).
    pub fn is_union_of_full_warpgroups(&self) -> bool {
        let touched = self.warps_touched();
        if touched.is_empty() {
            return false;
        }
        let groups: std::collections::BTreeSet<u32> = touched.iter().map(|w| w / 4).collect();
        groups
            .iter()
            .flat_map(|g| g * 4..g * 4 + 4)
            .all(|w| w < self.num_warps && self.is_full_warp(w))
    }

    /// The single selected thread, if the set has exactly one.
    pub fn single_thread(&self) -> Option<(u32, u32)> {
        let mut found = None;
        for w in 0..self.num_warps {
            for l in 0..32 {
                if self.contains(w, l) {
                    if found.is_some() {
                        return None;
                    }
                    found = Some((w, l));
                }
            }
        }
        found
    }
}

/// Result of statically evaluating an `If` predicate over one CTA.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ThreadFilter {
    Known(ThreadSet),
    Unknown,
}

impl ThreadFilter {
    pub fn known(&self) -> Option<&ThreadSet> {
        match self {
            ThreadFilter::Known(set) => Some(set),
            ThreadFilter::Unknown => None,
        }
    }
}

#[derive(Clone, Copy)]
enum Tri {
    Known(i64),
    Unknown,
}

fn eval_tri(value: &ScalarValue, warp: i64, lane: i64) -> Tri {
    match value {
        ScalarValue::Int(n) => Tri::Known(*n),
        ScalarValue::Var(_) => Tri::Unknown,
        ScalarValue::Scope(kind) => match kind {
            ScopeValueKind::TidInWg => Tri::Known((warp % 4) * 32 + lane),
            ScopeValueKind::LaneId => Tri::Known(lane),
            ScopeValueKind::WarpId => Tri::Known(warp),
            ScopeValueKind::WarpgroupId => Tri::Known(warp / 4),
            ScopeValueKind::CtaidInCluster | ScopeValueKind::CtaId | ScopeValueKind::NvshmemMyPe => {
                Tri::Unknown
            }
        },
        ScalarValue::Expr(e) => {
            let args: Vec<Tri> = e.args.iter().map(|a| eval_tri(a, warp, lane)).collect();
            // Absorption: resolve without the unknown side where the op allows it.
            match e.op {
                ScalarOp::And | ScalarOp::Mul => {
                    if args.iter().any(|a| matches!(a, Tri::Known(0))) {
                        return Tri::Known(0);
                    }
                }
                ScalarOp::Select => {
                    if let Tri::Known(c) = args[0] {
                        return args[if c != 0 { 1 } else { 2 }];
                    }
                }
                _ => {}
            }
            let mut known = Vec::with_capacity(args.len());
            for a in &args {
                match a {
                    Tri::Known(v) => known.push(*v),
                    Tri::Unknown => return Tri::Unknown,
                }
            }
            match apply_scalar_op(e.op, &known) {
                Ok(v) => Tri::Known(v),
                // A statically-detectable div/mod-by-zero is the runtime's
                // error to report; the filter just declines to predict.
                Err(_) => Tri::Unknown,
            }
        }
    }
}

/// Evaluate `cond` at every (warp, lane) of a CTA with `num_warps` warps.
pub fn static_thread_filter(cond: &ScalarValue, num_warps: u32) -> ThreadFilter {
    let mut bits = vec![false; (num_warps * 32) as usize];
    for warp in 0..num_warps {
        for lane in 0..32u32 {
            match eval_tri(cond, warp as i64, lane as i64) {
                Tri::Known(v) => bits[(warp * 32 + lane) as usize] = v != 0,
                Tri::Unknown => return ThreadFilter::Unknown,
            }
        }
    }
    ThreadFilter::Known(ThreadSet { num_warps, bits })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::dtype::{ScalarDType, VarBinding};
    use crate::ir::scalar::{Var, VarId};

    fn scope(kind: ScopeValueKind) -> ScalarValue {
        ScalarValue::Scope(kind)
    }
    fn eq(a: ScalarValue, b: i64) -> ScalarValue {
        ScalarValue::expr(ScalarOp::Eq, vec![a, ScalarValue::Int(b)])
    }
    fn and(a: ScalarValue, b: ScalarValue) -> ScalarValue {
        ScalarValue::expr(ScalarOp::And, vec![a, b])
    }
    fn runtime_var() -> ScalarValue {
        ScalarValue::Var(Var {
            id: VarId(7),
            binding: VarBinding::Scalar,
            dtype: ScalarDType::I32,
        })
    }

    #[test]
    fn warp_eq_selects_one_full_warp() {
        let f = static_thread_filter(&eq(scope(ScopeValueKind::WarpId), 3), 8);
        let set = f.known().unwrap();
        assert_eq!(set.is_exactly_one_full_warp(), Some(3));
        assert_eq!(set.count(), 32);
    }

    #[test]
    fn lane_eq_selects_lane0_of_every_warp() {
        let f = static_thread_filter(&eq(scope(ScopeValueKind::LaneId), 0), 4);
        let set = f.known().unwrap();
        assert_eq!(set.count(), 4);
        assert!(set.contains(2, 0));
        assert!(!set.contains(2, 1));
        assert!(set.is_exactly_one_full_warp().is_none());
    }

    #[test]
    fn warpgroup_eq_selects_full_warpgroup() {
        let f = static_thread_filter(&eq(scope(ScopeValueKind::WarpgroupId), 1), 12);
        let set = f.known().unwrap();
        assert_eq!(set.is_exactly_one_full_warpgroup(), Some(1));
        assert!(set.is_union_of_full_warpgroups());
        assert_eq!(set.warps_touched(), vec![4, 5, 6, 7]);
    }

    #[test]
    fn warpgroup_and_tid0_selects_single_thread() {
        let cond = and(
            eq(scope(ScopeValueKind::WarpgroupId), 1),
            eq(scope(ScopeValueKind::TidInWg), 0),
        );
        let f = static_thread_filter(&cond, 8);
        let set = f.known().unwrap();
        assert_eq!(set.single_thread(), Some((4, 0)));
    }

    #[test]
    fn runtime_var_is_unknown() {
        assert_eq!(
            static_thread_filter(&eq(runtime_var(), 0), 4),
            ThreadFilter::Unknown
        );
        // CTA coordinates are runtime-dependent too.
        assert_eq!(
            static_thread_filter(&eq(scope(ScopeValueKind::CtaId), 0), 4),
            ThreadFilter::Unknown
        );
    }

    #[test]
    fn and_absorption_resolves_mixed_predicates_to_false() {
        // warp_id == 7 is false everywhere in a 4-warp CTA, so the runtime
        // side never needs evaluating: Known(empty).
        let cond = and(eq(scope(ScopeValueKind::WarpId), 7), runtime_var());
        let set_filter = static_thread_filter(&cond, 4);
        let set = set_filter.known().unwrap();
        assert!(set.is_empty());
        // But if some thread satisfies the static side, the filter degrades.
        let cond = and(eq(scope(ScopeValueKind::WarpId), 2), runtime_var());
        assert_eq!(static_thread_filter(&cond, 4), ThreadFilter::Unknown);
    }

    #[test]
    fn modulo_shapes_work() {
        // warp_id % 4 == 0 -> warps 0 and 4 in an 8-warp CTA.
        let cond = eq(
            ScalarValue::expr(
                ScalarOp::Mod,
                vec![scope(ScopeValueKind::WarpId), ScalarValue::Int(4)],
            ),
            0,
        );
        let set_filter = static_thread_filter(&cond, 8);
        let set = set_filter.known().unwrap();
        assert_eq!(set.full_warps(), vec![0, 4]);
        assert!(!set.is_union_of_full_warpgroups());
    }

    #[test]
    fn intersection_narrows() {
        let wg = static_thread_filter(&eq(scope(ScopeValueKind::WarpgroupId), 0), 8);
        let lane0 = static_thread_filter(&eq(scope(ScopeValueKind::LaneId), 0), 8);
        let both = wg.known().unwrap().intersect(lane0.known().unwrap());
        assert_eq!(both.count(), 4);
        assert_eq!(both.warps_touched(), vec![0, 1, 2, 3]);
    }
}
