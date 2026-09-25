//! Temporal sheaf of executions (spec §5).
//!
//! The time site `T` is the poset of finite half-open intervals
//! `[start, end)` under inclusion. A run is a presheaf
//! `Run: T^op → Set` (truncated Spaces). Restriction is filtering events
//! to a subinterval. Descent: compatible local sections on a cover glue
//! to a unique global section; incompatible overlaps are reported.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use catnn_core::{CatError, CatResult};

/// A finite execution interval. Times are monotonic nanoseconds from an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Interval {
    pub start: u64,
    pub end: u64,
}

impl Interval {
    pub fn new(start: u64, end: u64) -> CatResult<Self> {
        if end < start {
            return Err(CatError::Interval(format!("[{start}, {end})")));
        }
        Ok(Self { start, end })
    }

    pub fn contains(self, t: u64) -> bool {
        t >= self.start && t < self.end
    }

    pub fn includes(self, other: Interval) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    pub fn intersect(self, other: Interval) -> Option<Interval> {
        let start = self.start.max(other.start);
        let end = self.end.min(other.end);
        if end > start {
            Some(Interval { start, end })
        } else {
            None
        }
    }

    pub fn duration_ns(self) -> u64 {
        self.end.saturating_sub(self.start)
    }
}

/// A single recorded event (local section data).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub t: u64,
    pub op: String,
    pub device: String,
    pub kind: EventKind,
    /// Payload checksum so overlapping traces can be compared.
    pub checksum: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKind {
    KernelStart,
    KernelEnd,
    TransferStart,
    TransferEnd,
}

/// A local section of `Run` over an interval, on one device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LocalSection {
    pub device: String,
    pub interval: Interval,
    pub events: Vec<Event>,
}

impl LocalSection {
    pub fn new(device: impl Into<String>, interval: Interval) -> Self {
        Self { device: device.into(), interval, events: vec![] }
    }

    pub fn push(&mut self, ev: Event) {
        self.events.push(ev);
    }

    /// Restriction functor `Run(I) → Run(J)` for `J ⊆ I`.
    pub fn restrict(&self, sub: Interval) -> CatResult<LocalSection> {
        if !self.interval.includes(sub) {
            return Err(CatError::Descent(format!(
                "cannot restrict [{}, {}) to [{}, {})",
                self.interval.start, self.interval.end, sub.start, sub.end
            )));
        }
        Ok(LocalSection {
            device: self.device.clone(),
            interval: sub,
            events: self.events.iter().filter(|e| sub.contains(e.t)).cloned().collect(),
        })
    }

    pub fn sorted(&self) -> Vec<Event> {
        let mut v = self.events.clone();
        v.sort_by_key(|e| (e.t, e.op.clone()));
        v
    }
}

/// Incompatibility on an overlap (failed descent).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DescentIssue {
    pub device_a: String,
    pub device_b: String,
    pub interval: Interval,
    pub reason: String,
}

/// A global section obtained by gluing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GlobalHistory {
    pub interval: Interval,
    pub events: Vec<Event>,
    pub issues: Vec<DescentIssue>,
}

impl GlobalHistory {
    pub fn is_coherent(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Two sections are compatible on `I ∩ J` when, for events that share
/// `(t, op, kind)`, checksums agree. Distinct devices may record the
/// same logical transfer from both sides; those must match.
pub fn compatible(a: &LocalSection, b: &LocalSection) -> Result<(), DescentIssue> {
    let Some(ov) = a.interval.intersect(b.interval) else {
        return Ok(());
    };
    let ra = a.restrict(ov).map_err(|e| DescentIssue {
        device_a: a.device.clone(),
        device_b: b.device.clone(),
        interval: ov,
        reason: e.to_string(),
    })?;
    let rb = b.restrict(ov).map_err(|e| DescentIssue {
        device_a: a.device.clone(),
        device_b: b.device.clone(),
        interval: ov,
        reason: e.to_string(),
    })?;
    let mut map: BTreeMap<(u64, String, String), u64> = BTreeMap::new();
    for e in &ra.events {
        map.insert((e.t, e.op.clone(), format!("{:?}", e.kind)), e.checksum);
    }
    for e in &rb.events {
        let key = (e.t, e.op.clone(), format!("{:?}", e.kind));
        if let Some(prev) = map.get(&key) {
            if *prev != e.checksum {
                return Err(DescentIssue {
                    device_a: a.device.clone(),
                    device_b: b.device.clone(),
                    interval: ov,
                    reason: format!("checksum mismatch on {} @ {} ({:?})", e.op, e.t, e.kind),
                });
            }
        }
    }
    Ok(())
}

/// Glue a cover of local sections. Events on disjoint regions are just
/// concatenated; overlaps must pass [`compatible`].
pub fn glue(sections: &[LocalSection]) -> CatResult<GlobalHistory> {
    if sections.is_empty() {
        return Ok(GlobalHistory {
            interval: Interval::new(0, 0)?,
            events: vec![],
            issues: vec![],
        });
    }
    let start = sections.iter().map(|s| s.interval.start).min().unwrap();
    let end = sections.iter().map(|s| s.interval.end).max().unwrap();
    let mut issues = Vec::new();
    for i in 0..sections.len() {
        for j in (i + 1)..sections.len() {
            if let Err(iss) = compatible(&sections[i], &sections[j]) {
                issues.push(iss);
            }
        }
    }
    let mut events: Vec<Event> = sections.iter().flat_map(|s| s.events.iter().cloned()).collect();
    events.sort_by_key(|e| (e.t, e.device.clone(), e.op.clone()));
    events.dedup_by(|a, b| {
        a.t == b.t && a.op == b.op && a.kind == b.kind && a.checksum == b.checksum
    });
    Ok(GlobalHistory { interval: Interval { start, end }, events, issues })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(t: u64, op: &str, dev: &str, sum: u64) -> Event {
        Event { t, op: op.into(), device: dev.into(), kind: EventKind::KernelEnd, checksum: sum }
    }

    #[test]
    fn restrict_filters() {
        let mut s = LocalSection::new("cpu", Interval::new(0, 100).unwrap());
        s.push(ev(10, "a", "cpu", 1));
        s.push(ev(50, "b", "cpu", 2));
        let r = s.restrict(Interval::new(0, 30).unwrap()).unwrap();
        assert_eq!(r.events.len(), 1);
    }

    #[test]
    fn glue_compatible() {
        let mut a = LocalSection::new("cpu", Interval::new(0, 50).unwrap());
        a.push(ev(10, "fc1", "cpu", 7));
        let mut b = LocalSection::new("gpu", Interval::new(40, 80).unwrap());
        b.push(ev(60, "fc2", "gpu", 8));
        let g = glue(&[a, b]).unwrap();
        assert!(g.is_coherent());
        assert_eq!(g.events.len(), 2);
    }

    #[test]
    fn glue_reports_incompatible_overlap() {
        let mut a = LocalSection::new("n0", Interval::new(0, 50).unwrap());
        a.push(ev(20, "xfer", "n0", 1));
        let mut b = LocalSection::new("n1", Interval::new(10, 60).unwrap());
        b.push(ev(20, "xfer", "n1", 2));
        let g = glue(&[a, b]).unwrap();
        assert!(!g.is_coherent());
    }
}
