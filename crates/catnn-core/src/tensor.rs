//! Tensor types are the *objects* of the symmetric monoidal category `Tens`.
//!
//! A morphism `f: A → B` in `Tens` is admissible only when the source and
//! target [`TensorType`]s (shape × dtype) match at every port. This is the
//! load-bearing type checker for sequential and monoidal composition.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{CatError, CatResult};

/// Element type of a tensor. Only `F32` is executable in the current backends;
/// the others exist so the SMC type checker can reject illegal wiring.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    F32,
    F16,
    I32,
    Bool,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 => 4,
            DType::F16 => 2,
            DType::Bool => 1,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::I32 => "i32",
            DType::Bool => "bool",
        }
    }

    pub fn parse(s: &str) -> CatResult<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(DType::F32),
            "f16" | "float16" => Ok(DType::F16),
            "i32" | "int32" => Ok(DType::I32),
            "bool" => Ok(DType::Bool),
            other => Err(CatError::Parse(format!("unknown dtype {other}"))),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Rank-static shape. Dynamic dims are out of scope; the SMC is strict.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: impl Into<Vec<usize>>) -> Self {
        Self { dims: dims.into() }
    }

    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn bytes(&self, dtype: DType) -> usize {
        self.numel() * dtype.size_bytes()
    }

    /// Matrix-multiply result shape: `[..., M, K] × [..., K, N] → [..., M, N]`.
    pub fn matmul(&self, rhs: &Shape) -> CatResult<Shape> {
        if self.rank() < 2 || rhs.rank() < 2 {
            return Err(CatError::ShapeMismatch(format!(
                "matmul needs rank ≥ 2, got {self} and {rhs}"
            )));
        }
        let (a_batch, a_m, a_k) = split_matrix(self);
        let (b_batch, b_k, b_n) = split_matrix(rhs);
        if a_k != b_k {
            return Err(CatError::ShapeMismatch(format!(
                "inner dim {a_k} vs {b_k} in {self} @ {rhs}"
            )));
        }
        if a_batch != b_batch {
            return Err(CatError::ShapeMismatch(format!(
                "batch {a_batch:?} vs {b_batch:?} in {self} @ {rhs}"
            )));
        }
        let mut dims = a_batch;
        dims.push(a_m);
        dims.push(b_n);
        Ok(Shape { dims })
    }

    pub fn broadcast_add(&self, other: &Shape) -> CatResult<Shape> {
        if self == other {
            return Ok(self.clone());
        }
        // Bias broadcast: [M, N] + [N] or [M, N] + [1, N]
        if self.rank() == 2 && other.rank() == 1 && self.dims[1] == other.dims[0] {
            return Ok(self.clone());
        }
        if self.rank() == 2
            && other.rank() == 2
            && other.dims[0] == 1
            && self.dims[1] == other.dims[1]
        {
            return Ok(self.clone());
        }
        Err(CatError::ShapeMismatch(format!(
            "cannot broadcast {self} + {other}"
        )))
    }

    pub fn last_dim(&self) -> CatResult<usize> {
        self.dims
            .last()
            .copied()
            .ok_or_else(|| CatError::ShapeMismatch("scalar has no last dim".into()))
    }
}

fn split_matrix(shape: &Shape) -> (Vec<usize>, usize, usize) {
    let n = shape.dims.len();
    let batch = shape.dims[..n - 2].to_vec();
    (batch, shape.dims[n - 2], shape.dims[n - 1])
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dims.is_empty() {
            return write!(f, "[]");
        }
        write!(f, "[")?;
        for (i, d) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            write!(f, "{d}")?;
        }
        write!(f, "]")
    }
}

/// An object of `Tens`: a single named tensor type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorType {
    pub shape: Shape,
    pub dtype: DType,
}

impl TensorType {
    pub fn new(shape: impl Into<Vec<usize>>, dtype: DType) -> Self {
        Self { shape: Shape::new(shape), dtype }
    }

    pub fn f32(shape: impl Into<Vec<usize>>) -> Self {
        Self::new(shape, DType::F32)
    }

    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    pub fn bytes(&self) -> usize {
        self.shape.bytes(self.dtype)
    }

    pub fn same_as(&self, other: &TensorType) -> CatResult<()> {
        if self.dtype != other.dtype {
            return Err(CatError::DTypeMismatch {
                left: self.dtype.to_string(),
                right: other.dtype.to_string(),
            });
        }
        if self.shape != other.shape {
            return Err(CatError::ShapeMismatch(format!(
                "{} vs {}",
                self.shape, other.shape
            )));
        }
        Ok(())
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.dtype, self.shape)
    }
}

/// Owned dense tensor used as runtime data. Storage is always `f32` host
/// memory; `dtype` records the intended semantic type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub ty: TensorType,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn zeros(ty: TensorType) -> Self {
        let n = ty.numel();
        Self { ty, data: vec![0.0; n] }
    }

    pub fn ones(ty: TensorType) -> Self {
        let n = ty.numel();
        Self { ty, data: vec![1.0; n] }
    }

    pub fn fill(ty: TensorType, value: f32) -> Self {
        let n = ty.numel();
        Self { ty, data: vec![value; n] }
    }

    pub fn from_vec(ty: TensorType, data: Vec<f32>) -> CatResult<Self> {
        if data.len() != ty.numel() {
            return Err(CatError::ShapeMismatch(format!(
                "data len {} != numel {}",
                data.len(),
                ty.numel()
            )));
        }
        Ok(Self { ty, data })
    }

    /// Deterministic pseudo-random fill (LCG) so examples and tests are stable.
    pub fn lcg(ty: TensorType, seed: u64) -> Self {
        let mut state = seed.max(1);
        let mut data = Vec::with_capacity(ty.numel());
        for _ in 0..ty.numel() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let unit = ((state >> 33) as u32) as f32 / (u32::MAX as f32);
            data.push(unit * 2.0 - 1.0);
        }
        Self { ty, data }
    }

    pub fn max_abs_diff(&self, other: &Tensor) -> CatResult<f32> {
        self.ty.same_as(&other.ty)?;
        let mut max = 0.0f32;
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            max = max.max((a - b).abs());
        }
        Ok(max)
    }

    pub fn checksum(&self) -> u64 {
        // FNV-1a over bit patterns; used by sheaf compatibility, not crypto.
        let mut h = 0xcbf29ce484222325u64;
        for x in &self.data {
            let bits = x.to_bits() as u64;
            h ^= bits;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_shape() {
        let a = Shape::new([4, 16]);
        let b = Shape::new([16, 8]);
        assert_eq!(a.matmul(&b).unwrap(), Shape::new([4, 8]));
    }

    #[test]
    fn matmul_rejects_inner_mismatch() {
        let a = Shape::new([4, 16]);
        let b = Shape::new([15, 8]);
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn tensor_lcg_stable() {
        let t = Tensor::lcg(TensorType::f32([2, 2]), 7);
        assert_eq!(t.data.len(), 4);
        let t2 = Tensor::lcg(TensorType::f32([2, 2]), 7);
        assert_eq!(t.data, t2.data);
    }
}
