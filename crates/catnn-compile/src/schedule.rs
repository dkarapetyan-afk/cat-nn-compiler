//! Unified training / inference scheduler.
//!
//! One timeline covers kernel **loads**, **launches**, buffer **alloc/free**,
//! transfers, prefetch/offload, and rematerialization. Liveness is computed
//! for both tensors and kernel modules. Buffers are placed by a first-fit
//! arena that aliases non-overlapping lifetimes. Peak occupancy per region
//! is the max over the simulated timeline — not a closed form.

use catnn_core::architecture::{
    DeviceMemory, MemoryBudget, MemoryHierarchy, MemoryRegionKind, RegionPeaks,
};
use catnn_core::category::TwoCell;
use catnn_core::{CatError, CatResult};
use serde::{Deserialize, Serialize};

use crate::kernel::resource::KernelResource;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventKind {
    AllocBuffer { id: String, bytes: u64, offset: u64, region: MemoryRegionKind },
    FreeBuffer { id: String },
    LoadKernel { name: String, bytes: u64 },
    EvictKernel { name: String },
    Launch { name: String, scratch: u64 },
    Transfer { id: String, bytes: u64, from: String, to: String },
    Recompute { id: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub t: f64,
    pub kind: EventKind,
    pub peaks: RegionPeaks,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArenaAlloc {
    pub id: String,
    pub bytes: u64,
    pub offset: u64,
    pub live_from: usize,
    pub live_to: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScheduledStep {
    pub feasible: bool,
    pub overflow_region: Option<MemoryRegionKind>,
    pub events: Vec<Event>,
    pub arena: Vec<ArenaAlloc>,
    pub peaks: RegionPeaks,
    pub cap: DeviceMemory,
    pub step_seconds: f64,
    pub recompute_overhead: f64,
    pub kernel_reloads: usize,
    pub cells: Vec<TwoCell>,
    pub min_budget: MemoryBudget,
    pub rationale: String,
}

#[derive(Debug, Clone)]
pub struct BufferReq {
    pub id: String,
    pub bytes: u64,
    /// Inclusive logical time of first use.
    pub live_from: usize,
    /// Inclusive logical time of last use.
    pub live_to: usize,
    pub checkpoint: bool,
}

#[derive(Debug, Clone)]
pub struct LaunchReq {
    pub at: usize,
    pub kernel: KernelResource,
}

/// First-fit arena: assign offsets so simultaneously live buffers do not overlap.
pub fn place_arena(reqs: &[BufferReq], cap: u64) -> CatResult<Vec<ArenaAlloc>> {
    let mut placed: Vec<ArenaAlloc> = Vec::new();
    let mut ordered: Vec<usize> = (0..reqs.len()).collect();
    ordered.sort_by_key(|&i| (reqs[i].live_from, std::cmp::Reverse(reqs[i].bytes)));
    for i in ordered {
        let r = &reqs[i];
        let mut forbidden: Vec<(u64, u64)> = placed
            .iter()
            .filter(|p| lifetimes_overlap(p.live_from, p.live_to, r.live_from, r.live_to))
            .map(|p| (p.offset, p.offset + p.bytes))
            .collect();
        forbidden.sort_by_key(|s| s.0);
        let mut cursor = 0u64;
        let mut chosen = None;
        for (s, e) in forbidden {
            if s >= cursor + r.bytes {
                chosen = Some(cursor);
                break;
            }
            cursor = cursor.max(e);
        }
        let off = chosen.unwrap_or(cursor);
        if off.saturating_add(r.bytes) > cap {
            return Err(CatError::Fibration(format!(
                "buffer arena overflow: {} needs {} B at t={}..{} (cap {cap})",
                r.id, r.bytes, r.live_from, r.live_to
            )));
        }
        placed.push(ArenaAlloc {
            id: r.id.clone(),
            bytes: r.bytes,
            offset: off,
            live_from: r.live_from,
            live_to: r.live_to,
        });
    }
    Ok(placed)
}

fn lifetimes_overlap(a0: usize, a1: usize, b0: usize, b1: usize) -> bool {
    a0 <= b1 && b0 <= a1
}

/// Assert arena offsets do not overlap for simultaneously live buffers.
pub fn arena_conflict(arena: &[ArenaAlloc]) -> Option<(String, String)> {
    for (i, a) in arena.iter().enumerate() {
        for b in arena.iter().skip(i + 1) {
            if lifetimes_overlap(a.live_from, a.live_to, b.live_from, b.live_to) {
                let a1 = a.offset + a.bytes;
                let b1 = b.offset + b.bytes;
                if a.offset < b1 && b.offset < a1 {
                    return Some((a.id.clone(), b.id.clone()));
                }
            }
        }
    }
    None
}

pub struct ScheduleRequest<'a> {
    pub buffers: &'a [BufferReq],
    pub launches: &'a [LaunchReq],
    pub regions: &'a DeviceMemory,
    pub hierarchy: &'a MemoryHierarchy,
    pub allow_recompute: bool,
    pub allow_reload: bool,
}

/// Build the fastest feasible schedule under the region caps, or name
/// the overflowing region and the minimum caps that would fit.
pub fn schedule(req: ScheduleRequest<'_>) -> ScheduledStep {
    let mut buffers: Vec<BufferReq> = req.buffers.to_vec();
    let mut cells = Vec::new();
    let mut recompute_overhead = 0.0;
    let mut kernel_reloads = 0usize;

    // Checkpoint the largest long-lived activation if the arena misses.
    let mut arena = match place_arena(&buffers, req.regions.buffer.bytes) {
        Ok(a) => a,
        Err(_) if req.allow_recompute => {
            if let Some((idx, _)) =
                buffers.iter().enumerate().filter(|(_, b)| !b.checkpoint).max_by_key(|(_, b)| {
                    b.bytes * (b.live_to.saturating_sub(b.live_from) as u64 + 1)
                })
            {
                let id = buffers[idx].id.clone();
                let bytes = buffers[idx].bytes;
                buffers[idx].checkpoint = true;
                // Rematerialize: live only at producer and consumer, not between.
                let t0 = buffers[idx].live_from;
                let t1 = buffers[idx].live_to;
                buffers[idx].live_to = t0;
                // A second copy at the consumer time (recomputed).
                buffers.push(BufferReq {
                    id: format!("{id}#recomp"),
                    bytes,
                    live_from: t1,
                    live_to: t1,
                    checkpoint: true,
                });
                recompute_overhead += bytes as f64 / req.regions.buffer.bandwidth_bps.max(1.0);
                cells.push(TwoCell::recompute(&id, "activation-checkpoint"));
                match place_arena(&buffers, req.regions.buffer.bytes) {
                    Ok(a) => a,
                    Err(_) => {
                        return infeasible(
                            req.regions,
                            MemoryRegionKind::Buffer,
                            &buffers,
                            &cells,
                            recompute_overhead,
                        );
                    }
                }
            } else {
                return infeasible(
                    req.regions,
                    MemoryRegionKind::Buffer,
                    &buffers,
                    &cells,
                    recompute_overhead,
                );
            }
        }
        Err(_) => {
            return infeasible(
                req.regions,
                MemoryRegionKind::Buffer,
                &buffers,
                &cells,
                recompute_overhead,
            );
        }
    };

    if arena_conflict(&arena).is_some() {
        return infeasible(
            req.regions,
            MemoryRegionKind::Buffer,
            &buffers,
            &cells,
            recompute_overhead,
        );
    }

    // Kernel residency with LRU eviction when the code region is tight.
    let mut events = Vec::new();
    let mut t = 0.0;
    let mut resident: Vec<(String, u64)> = Vec::new();
    let mut code_live = 0u64;
    let mut peaks = RegionPeaks::default();
    let mut buffer_live = 0u64;
    let mut scratch_live = 0u64;

    let mut launches: Vec<&LaunchReq> = req.launches.iter().collect();
    launches.sort_by_key(|l| l.at);

    let mut next_buf = 0usize;
    let mut sorted_bufs = buffers.clone();
    sorted_bufs.sort_by_key(|b| b.live_from);

    let max_t = launches.iter().map(|l| l.at).max().unwrap_or(0).max(
        buffers.iter().map(|b| b.live_to).max().unwrap_or(0),
    );

    for tick in 0..=max_t {
        while next_buf < sorted_bufs.len() && sorted_bufs[next_buf].live_from == tick {
            let b = &sorted_bufs[next_buf];
            let off = arena.iter().find(|a| a.id == b.id).map(|a| a.offset).unwrap_or(0);
            buffer_live += b.bytes;
            let p = RegionPeaks { code: code_live, buffer: buffer_live, scratch: scratch_live };
            peaks = peaks.max_with(&p);
            events.push(Event {
                t,
                kind: EventKind::AllocBuffer {
                    id: b.id.clone(),
                    bytes: b.bytes,
                    offset: off,
                    region: MemoryRegionKind::Buffer,
                },
                peaks: p,
            });
            next_buf += 1;
        }

        for launch in launches.iter().filter(|l| l.at == tick) {
            let k = &launch.kernel;
            let need = k.code_footprint();
            if !resident.iter().any(|(n, _)| n == &k.name) {
                while code_live + need > req.regions.code.bytes && !resident.is_empty() && req.allow_reload
                {
                    let (evict, sz) = resident.remove(0);
                    code_live = code_live.saturating_sub(sz);
                    kernel_reloads += 1;
                    cells.push(TwoCell::kernel_residency(&evict, false));
                    // Alternative 2-cell: fuse the evicted module with the
                    // incoming one to cut the code-region footprint.
                    cells.push(TwoCell::fuse_code(vec![evict.clone(), k.name.clone()]));
                    let p = RegionPeaks { code: code_live, buffer: buffer_live, scratch: scratch_live };
                    events.push(Event {
                        t,
                        kind: EventKind::EvictKernel { name: evict },
                        peaks: p,
                    });
                    t += req.regions.code.latency_s;
                }
                if code_live + need > req.regions.code.bytes {
                    return infeasible(
                        req.regions,
                        MemoryRegionKind::Code,
                        &buffers,
                        &cells,
                        recompute_overhead,
                    );
                }
                code_live += need;
                resident.push((k.name.clone(), need));
                t += req.regions.code.transfer_seconds(need);
                let p = RegionPeaks { code: code_live, buffer: buffer_live, scratch: scratch_live };
                peaks = peaks.max_with(&p);
                events.push(Event {
                    t,
                    kind: EventKind::LoadKernel { name: k.name.clone(), bytes: need },
                    peaks: p,
                });
            }
            let scratch = k.scratch_bytes();
            if scratch > req.regions.scratch.bytes {
                return infeasible(
                    req.regions,
                    MemoryRegionKind::Scratch,
                    &buffers,
                    &cells,
                    recompute_overhead,
                );
            }
            let p = RegionPeaks { code: code_live, buffer: buffer_live, scratch };
            peaks = peaks.max_with(&p);
            events.push(Event {
                t,
                kind: EventKind::Launch { name: k.name.clone(), scratch },
                peaks: p,
            });
            scratch_live = scratch;
            t += 1e-6 + k.code_bytes as f64 / req.regions.buffer.bandwidth_bps.max(1.0) * 0.01;
        }

        for b in buffers.iter().filter(|b| b.live_to == tick) {
            buffer_live = buffer_live.saturating_sub(b.bytes);
            events.push(Event {
                t,
                kind: EventKind::FreeBuffer { id: b.id.clone() },
                peaks: RegionPeaks { code: code_live, buffer: buffer_live, scratch: scratch_live },
            });
        }
    }

    let _ = req.hierarchy;
    let min_budget = MemoryBudget {
        code: Some(peaks.code.max(1)),
        buffer: peaks.buffer.max(1),
        scratch: Some(peaks.scratch.max(1)),
    };
    let rationale = format!(
        "simulated peaks code={:.1} KiB buffer={:.1} MiB scratch={} B; \
         reloads={kernel_reloads} recompute={:.2e}s; {} events",
        peaks.code as f64 / 1024.0,
        peaks.buffer as f64 / (1 << 20) as f64,
        peaks.scratch,
        recompute_overhead,
        events.len()
    );
    ScheduledStep {
        feasible: true,
        overflow_region: None,
        events,
        arena: std::mem::take(&mut arena),
        peaks,
        cap: req.regions.clone(),
        step_seconds: t + recompute_overhead,
        recompute_overhead,
        kernel_reloads,
        cells,
        min_budget,
        rationale,
    }
}

fn infeasible(
    regions: &DeviceMemory,
    overflow: MemoryRegionKind,
    buffers: &[BufferReq],
    cells: &[TwoCell],
    recompute_overhead: f64,
) -> ScheduledStep {
    let buf = buffers.iter().map(|b| b.bytes).sum::<u64>();
    ScheduledStep {
        feasible: false,
        overflow_region: Some(overflow),
        events: vec![],
        arena: vec![],
        peaks: RegionPeaks { code: 0, buffer: buf, scratch: 0 },
        cap: regions.clone(),
        step_seconds: f64::INFINITY,
        recompute_overhead,
        kernel_reloads: 0,
        cells: cells.to_vec(),
        min_budget: MemoryBudget {
            code: Some(regions.code.bytes),
            buffer: buf.max(regions.buffer.bytes),
            scratch: Some(regions.scratch.bytes),
        },
        rationale: match overflow {
            MemoryRegionKind::Code => format!(
                "infeasible: code region overflow (cap {} B) — too many distinct \
                 kernel modules resident (expert / op binaries)",
                regions.code.bytes
            ),
            MemoryRegionKind::Buffer => format!(
                "infeasible: buffer region overflow (cap {} B) — live tensors \
                 exceed the arena",
                regions.buffer.bytes
            ),
            MemoryRegionKind::Scratch => format!(
                "infeasible: scratch region overflow (cap {} B) — a launch's \
                 SMEM/register occupancy does not fit",
                regions.scratch.bytes
            ),
        },
    }
}

/// Fastest feasible schedule under a structured budget `M`.
pub fn fastest_under(req: ScheduleRequest<'_>, budget: MemoryBudget) -> ScheduledStep {
    let mut regions = req.regions.clone();
    regions.apply_budget(&budget);
    schedule(ScheduleRequest {
        buffers: req.buffers,
        launches: req.launches,
        regions: &regions,
        hierarchy: req.hierarchy,
        allow_recompute: req.allow_recompute,
        allow_reload: req.allow_reload,
    })
}

/// Minimum capacity of one region that still admits a feasible schedule,
/// holding the other two caps fixed. Returns the overflowing region when
/// even the original cap cannot fit.
pub fn min_region(
    req: ScheduleRequest<'_>,
    kind: MemoryRegionKind,
) -> Result<u64, MemoryRegionKind> {
    let mut lo = 1u64;
    let mut hi = req.regions.region(kind).bytes.max(1);
    let mut best = None;
    while lo <= hi {
        let mid = lo + (hi - lo) / 2;
        let mut regions = req.regions.clone();
        match kind {
            MemoryRegionKind::Code => regions.code.bytes = mid,
            MemoryRegionKind::Buffer => regions.buffer.bytes = mid,
            MemoryRegionKind::Scratch => regions.scratch.bytes = mid,
        }
        let s = schedule(ScheduleRequest {
            buffers: req.buffers,
            launches: req.launches,
            regions: &regions,
            hierarchy: req.hierarchy,
            allow_recompute: req.allow_recompute,
            allow_reload: req.allow_reload,
        });
        if s.feasible {
            best = Some(mid);
            hi = mid.saturating_sub(1);
        } else {
            lo = mid.saturating_add(1);
        }
    }
    best.ok_or(kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use catnn_core::architecture::KernelBackend;
    use catnn_core::category::OpenOp;
    use crate::kernel::resource::describe;

    fn dummy_k(name: &str) -> KernelResource {
        describe(&OpenOp::Relu, name, KernelBackend::CpuRust)
    }

    #[test]
    fn arena_aliases_non_overlapping() {
        let reqs = vec![
            BufferReq { id: "a".into(), bytes: 8, live_from: 0, live_to: 1, checkpoint: false },
            BufferReq { id: "b".into(), bytes: 8, live_from: 2, live_to: 3, checkpoint: false },
        ];
        let a = place_arena(&reqs, 8).unwrap();
        assert!(arena_conflict(&a).is_none());
        assert!(a.iter().map(|x| x.offset).max().unwrap_or(0) + 8 <= 16);
        // Both can sit at offset 0.
        assert!(a.iter().all(|x| x.offset == 0));
    }

    #[test]
    fn arena_overlap_detected() {
        let bad = vec![
            ArenaAlloc { id: "a".into(), bytes: 8, offset: 0, live_from: 0, live_to: 2 },
            ArenaAlloc { id: "b".into(), bytes: 8, offset: 4, live_from: 1, live_to: 3 },
        ];
        assert!(arena_conflict(&bad).is_some());
    }

    #[test]
    fn simulated_peak_respects_budget() {
        let regions = DeviceMemory::desktop_cpu();
        let hier = MemoryHierarchy::desktop_cpu();
        let bufs = vec![BufferReq {
            id: "x".into(),
            bytes: 1024,
            live_from: 0,
            live_to: 0,
            checkpoint: false,
        }];
        let k = dummy_k("relu");
        let launches = vec![LaunchReq { at: 0, kernel: k }];
        let s = schedule(ScheduleRequest {
            buffers: &bufs,
            launches: &launches,
            regions: &regions,
            hierarchy: &hier,
            allow_recompute: true,
            allow_reload: true,
        });
        assert!(s.feasible);
        assert!(s.peaks.exceeds(&regions).is_none());
        assert!(arena_conflict(&s.arena).is_none());
    }

    #[test]
    fn tight_code_causes_reloads() {
        let mut regions = DeviceMemory::desktop_cpu();
        let k1 = dummy_k("relu0");
        let k2 = dummy_k("relu1");
        // Cap code so two kernels cannot stay resident.
        regions.code.bytes = k1.code_footprint() + 32;
        let hier = MemoryHierarchy::desktop_cpu();
        let launches = vec![
            LaunchReq { at: 0, kernel: k1 },
            LaunchReq { at: 1, kernel: k2 },
            LaunchReq { at: 2, kernel: dummy_k("relu0") },
        ];
        let s = schedule(ScheduleRequest {
            buffers: &[],
            launches: &launches,
            regions: &regions,
            hierarchy: &hier,
            allow_recompute: true,
            allow_reload: true,
        });
        assert!(s.feasible, "{}", s.rationale);
        assert!(s.kernel_reloads >= 1, "reloads={}", s.kernel_reloads);
        assert!(s.peaks.code <= regions.code.bytes);
        assert!(s.cells.iter().any(|c| matches!(
            c.rewrite,
            catnn_core::category::Rewrite::KernelResidency { .. }
        )));
        assert!(s.cells.iter().any(|c| matches!(
            c.rewrite,
            catnn_core::category::Rewrite::FuseCode { .. }
        )));
    }

    #[test]
    fn code_overflow_names_code_region() {
        let mut regions = DeviceMemory::desktop_cpu();
        regions.code.bytes = 8;
        let hier = MemoryHierarchy::desktop_cpu();
        let launches = vec![LaunchReq { at: 0, kernel: dummy_k("relu") }];
        let s = schedule(ScheduleRequest {
            buffers: &[],
            launches: &launches,
            regions: &regions,
            hierarchy: &hier,
            allow_recompute: true,
            allow_reload: true,
        });
        assert!(!s.feasible);
        assert_eq!(s.overflow_region, Some(MemoryRegionKind::Code));
        assert!(s.rationale.contains("code region"));
    }

    #[test]
    fn shrinking_scratch_names_scratch_and_keeps_other_caps() {
        let mut regions = DeviceMemory::desktop_cpu();
        let k = {
            let mut k = describe(
                &OpenOp::Linear { in_features: 16, out_features: 16, bias: false },
                "fc",
                KernelBackend::CpuRust,
            );
            k.static_smem_bytes = 4096;
            k
        };
        regions.scratch.bytes = 128;
        let hier = MemoryHierarchy::desktop_cpu();
        let launches = vec![LaunchReq { at: 0, kernel: k }];
        let s = schedule(ScheduleRequest {
            buffers: &[],
            launches: &launches,
            regions: &regions,
            hierarchy: &hier,
            allow_recompute: true,
            allow_reload: true,
        });
        assert!(!s.feasible);
        assert_eq!(s.overflow_region, Some(MemoryRegionKind::Scratch));
        assert!(s.rationale.contains("scratch"));
    }

    #[test]
    fn min_region_code_is_at_most_unconstrained_peak() {
        let regions = DeviceMemory::desktop_cpu();
        let hier = MemoryHierarchy::desktop_cpu();
        let launches = vec![
            LaunchReq { at: 0, kernel: dummy_k("a") },
            LaunchReq { at: 1, kernel: dummy_k("b") },
        ];
        let req = ScheduleRequest {
            buffers: &[],
            launches: &launches,
            regions: &regions,
            hierarchy: &hier,
            allow_recompute: true,
            allow_reload: true,
        };
        let unconstrained = schedule(req);
        let req = ScheduleRequest {
            buffers: &[],
            launches: &launches,
            regions: &regions,
            hierarchy: &hier,
            allow_recompute: true,
            allow_reload: true,
        };
        let min_c = min_region(req, MemoryRegionKind::Code).unwrap();
        assert!(min_c <= unconstrained.peaks.code.max(1));
        assert!(min_c > 0);
    }
}
