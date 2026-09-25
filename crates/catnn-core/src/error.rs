use thiserror::Error;

/// Type and structure errors raised by the categorical constructions.
///
/// Composition in the tensor SMC, lens laws, and fibration reindexing
/// all return this type instead of panicking: a failed compose is a
/// failed morphism, not an informal crash.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CatError {
    #[error("interface mismatch: expected {expected}, got {actual}")]
    InterfaceMismatch { expected: String, actual: String },

    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("dtype mismatch: {left} vs {right}")]
    DTypeMismatch { left: String, right: String },

    #[error("unknown port {port} on interface {interface}")]
    UnknownPort { interface: String, port: String },

    #[error("unknown component {0}")]
    UnknownComponent(String),

    #[error("unknown architecture {0}")]
    UnknownArchitecture(String),

    #[error("unknown realization {0}")]
    UnknownRealization(String),

    #[error("wiring error: {0}")]
    Wiring(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("fibration error: {0}")]
    Fibration(String),

    #[error("equivalence witness failed: {0}")]
    Witness(String),

    #[error("sheaf descent failed: {0}")]
    Descent(String),

    #[error("invalid interval: {0}")]
    Interval(String),

    #[error("illegal optic used as a lawful lens: {0}")]
    Unlawful(String),

    #[error("io: {0}")]
    Io(String),
}

pub type CatResult<T> = Result<T, CatError>;
