//! Per-device runtime handlers. A handler owns a mailbox and executes
//! kernels assigned to its device.

use std::time::Instant;

use catnn_compile::ir::IrNode;
use catnn_compile::kernel;
use catnn_core::architecture::Device;
use catnn_core::tensor::Tensor;
use catnn_core::CatResult;

use crate::sheaf::{Event, EventKind, Interval, LocalSection};

pub struct DeviceHandler {
    pub device: Device,
    pub section: LocalSection,
    epoch: Instant,
}

impl DeviceHandler {
    pub fn new(device: Device) -> Self {
        let epoch = Instant::now();
        Self {
            section: LocalSection::new(&device.id, Interval { start: 0, end: 1 }),
            device,
            epoch,
        }
    }

    fn now_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }

    pub fn run_kernel(&mut self, node: &IrNode, inputs: &[Tensor]) -> CatResult<Tensor> {
        let t0 = self.now_ns();
        self.section.push(Event {
            t: t0,
            op: node.name.clone(),
            device: self.device.id.clone(),
            kind: EventKind::KernelStart,
            checksum: 0,
        });
        // GPU-placed ops still execute the CPU kernel here: the WGSL was
        // generated and validated at compile time. We do not pretend a
        // GPU ran. See README.
        let y = kernel::eval_cpu(node, inputs)?;
        let t1 = self.now_ns().max(t0 + 1);
        self.section.push(Event {
            t: t1,
            op: node.name.clone(),
            device: self.device.id.clone(),
            kind: EventKind::KernelEnd,
            checksum: y.checksum(),
        });
        self.section.interval.end = self.section.interval.end.max(t1 + 1);
        Ok(y)
    }

    /// Record a kernel span without executing it. The adjoint is a separate
    /// Rust function from the forward kernel; the span is the schedule entry.
    pub fn note_kernel(&mut self, name: &str, checksum: u64) {
        let t0 = self.now_ns();
        self.section.push(Event {
            t: t0,
            op: name.to_string(),
            device: self.device.id.clone(),
            kind: EventKind::KernelStart,
            checksum: 0,
        });
        let t1 = self.now_ns().max(t0 + 1);
        self.section.push(Event {
            t: t1,
            op: name.to_string(),
            device: self.device.id.clone(),
            kind: EventKind::KernelEnd,
            checksum,
        });
        self.section.interval.end = self.section.interval.end.max(t1 + 1);
    }

    pub fn record_transfer(&mut self, tag: &str, bytes_checksum: u64, start: bool) {
        let t = self.now_ns();
        self.section.push(Event {
            t,
            op: tag.to_string(),
            device: self.device.id.clone(),
            kind: if start { EventKind::TransferStart } else { EventKind::TransferEnd },
            checksum: bytes_checksum,
        });
        self.section.interval.end = self.section.interval.end.max(t + 1);
    }
}
