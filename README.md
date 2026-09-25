# catnn

A Rust framework that treats a neural network as an **open system in a
symmetric monoidal category**, compiles it across a heterogeneous cluster,
records execution as a **temporal sheaf**, and adapts the realization with
an **optic** (not a lens). The categorical constructions are the type system
and the runtime, not decoration.

The design follows the attached note
`uploads/categorical-nn-spec.md` (copied conceptually below). ∞-coherence is
truncated to 2-cells with *checkable* numerical witnesses — that approximation
is explicit in the types.

## What you can do

```bash
# built-in examples
cargo run -p catnn-cli -- compile mlp --cluster demo
cargo run -p catnn-cli -- run mlp --cluster demo
cargo run -p catnn-cli -- adapt mlp --cluster demo
cargo run -p catnn-cli -- inspect transformer --cluster demo

# textual specs (TOML, JSON, or .catnn)
cargo run -p catnn-cli -- run examples/mlp.catnn --cluster examples/cluster.toml
cargo run -p catnn-cli -- run examples/transformer.catnn --cluster demo
```

`cargo test --workspace` and `cargo clippy --workspace -- -D warnings` are
expected to pass.

## Workspace

| Crate | Role |
|---|---|
| `crates/catnn-core` | SMC of tensor interfaces, lenses / optics, open systems, cospans, architecture category `A`, hardware category `H`, fibration `R`, spec DSL + parsers |
| `crates/catnn-compile` | Lowering `A → IR`, placement on `H`, CPU kernels (Rayon + `wide` SIMD), WGSL generation + naga validation, naïve reference |
| `crates/catnn-rt` | Device handlers, cluster scheduler (threads + channels, optional TCP), temporal sheaf, adaptive optic |
| `crates/catnn-cli` | `catnn compile \| run \| adapt \| inspect` |

## Mapping: mathematics → Rust

Numbers refer to sections of the design document.

| Construction | Where it lives |
|---|---|
| **§1 Classical lens** `get/put` with PutGet, GetPut, PutPut | `catnn_core::lens::lawful::LawfulLens`. Laws are methods that return `Err` on failure. Property-tested on `first_of_pair` and `identity_f64`. |
| **§1 Higher lenses** | Truncated: we do not store 2-dimensional coherences of Capriotti–Danielsson–Vezzosi. The three 1-laws are the checkable fragment. |
| **§2 Indexed / dependent lenses, `R: A^op → Cat`** | `catnn_core::architecture::fibration`. `Fibre` is `R(A)`; `TotalObj` is an object of `∫R`; `reindex` is pullback along an `ArchMorphism`. `reindex_id_is_id` is the identity law of the functor. |
| **§2 Grothendieck / cartesian fibration** | Same module. Morphisms of `A` are refinements (`Id`, `Fuse`, `Lower`, `Repartition`). |
| **§3 AISys objects = interfaces** | `catnn_core::category::Interface` — finite products of named `TensorType`s. |
| **§3 1-morphisms = open components** | `Hom` / `OpenComponent`. Sequential composition is `Tens::compose` (type-checked); parallel is `Tens::tensor`. |
| **§3 2-morphisms = refinements** | `TwoCell` + `Rewrite` (`Fuse`, `Lower`, `Reassign`, `Partition`, `Refl`) carrying an `EquivWitness` (absolute tolerance, seeds). Verified by running CPU vs reference. |
| **§3 3-morphisms / ∞-coherence** | **Not implemented.** Equality of 2-cells is not a mapping space. Documented gap. |
| **§4 Open dynamical systems** | `OpenComponent { state, input, output, op, moore }`. Neural layers are Mealy (`readout` sees `U`); Moore is the flag. Sequential composition is `then` (state becomes `S⊗S'`). |
| **§4 Wiring diagrams / structured cospans** | `StructuredCospan`: `left → apex ← right`. `compose` is a port-identifying pushout of apex graphs. |
| **§5 Time site, `Run: T^op → Set`** | `catnn_rt::sheaf::{Interval, LocalSection}`. `restrict` is the restriction map. |
| **§5 Descent / gluing** | `glue` + `compatible`. Overlaps with disagreeing checksums become `DescentIssue`s on `GlobalHistory`. |
| **§6 `H` = hardware category** | `Cluster`, `Node`, `Device`, `Link`. Transfer cost is `latency + bytes/bandwidth`. |
| **§6 Runtime graph = diagram in AISys** | `IrGraph` (the compile image of the architecture) plus `Realization.placement`. |
| **§7 Architecture ↔ runtime as a lens** | Compile is `catnn_compile::compile : A → R`. Observe is `RunResult.observation`. **Update is not a lens.** |
| **§7 Unlawful optic with residual** | `catnn_core::lens::optic::{Optic, PolicyState, UnlawfulWitness}`. The optimizer is this optic; `UnlawfulWitness` names the failed law (`PutGet`). |
| **§8 Adaptive loop** | `catnn_rt::adapt::adapt_once`. Fuse linear+ReLU on the same device and/or move a heavy op onto the fastest device when the cost model improves. One 2-cell is recorded per step. |
| **§9 Equivalence, not equality** | `EquivWitness.accept(max_abs)`. CPU kernels are allowed to differ from the naïve reference inside a tolerance. |
| **§10 Four levels** | `Architecture` (`A`); `Realization` / `Fibre` (`R → A`); `LocalSection` / `GlobalHistory` (`Run`); `Optic` + `TwoCell` (`AISys` 2-cells). The central square commutes only up to the recorded 2-cell. |

## Spec layer

Three surface syntaxes parse into the same `NetworkSpec` and then into an
`Architecture`:

1. **Rust DSL** — `NetworkBuilder::new("mlp", 4).linear("fc1", 16, 32)?.relu("act")?...`. Composition returns `CatResult`, so a shape error is a failed morphism.
2. **TOML / JSON** — tagged `[[layers]]` (see `examples/mlp.toml`).
3. **`.catnn`** — line-oriented (`examples/mlp.catnn`, `examples/transformer.catnn`).

The SMC `Tens` rejects sequential composition unless every port agrees in
shape and dtype.

## Compiler and backends

`compile(arch, cluster)`:

1. Flattens the hom tree to an `IrGraph` (explicit input node + one node per component).
2. Places ops with a greedy cost model, then a local swap search.
3. Enumerates kernel backends: `CpuRust` on CPU devices, `Wgsl` on GPU devices.
4. Emits WGSL for every GPU-placed op and **validates it with naga**.

### What actually executes

| Backend | Status |
|---|---|
| **CPU (Rust + Rayon + `wide` f32x8 SIMD matmul)** | **Executes.** This is the runtime path for every op, including GPU-placed ones. |
| **Naïve reference** | **Executes**, used as the left-hand side of every numerical 2-cell. |
| **WGSL / GPU** | **Generated and naga-validated. Not dispatched to a GPU in this environment.** There is no wgpu device in the default CI VM; we refuse to fake a GPU run. GPU placement still *schedules* the op on a GPU handler and records transfers, then the CPU kernel runs (logged as “CPU fallback after WGSL validation”). |

To attach a real GPU later: construct a wgpu device, load the already-generated
`WgslKernel.source`, and replace `DeviceHandler::run_kernel` for `DeviceKind::Gpu`.

## Runtime

- One `DeviceHandler` per device, each accumulating a `LocalSection`.
- Cross-device tensors go through `ChannelFabric` (std `mpsc`). Node threads
  are spawned so the cluster is concurrent, not a single-threaded walk.
- `transfer::spawn_tcp_node` / `send_tcp` is a length-prefixed localhost
  protocol for a real multi-process stand-in; the default `run` path uses
  channels.
- After execution, local sections are glued. Incompatible overlaps fail
  descent (see `sheaf` tests).

## Adaptive loop

```
A  --compile-->  R  --run-->  τ  --observe-->  O  --optic-->  (A', P')
```

`P` is `PolicyState` (step, thresholds, last latency, residual note). The
optic will

- record a fusion group for `Linear; Relu` on the same device, and/or
- re-place the last heavy op onto the highest-GFLOP device when the *estimated*
  cost drops by `replace_threshold`.

Improvement is measured by the cost model and by observed latency. Numerics
are re-checked after the step. This is **not** a lawful lens: the new
observation is not the old one (`PutGet` fails), which is why the type is
`Optic` and why `UnlawfulWitness` is stored on every `AdaptStep`.

## Approximations and gaps

Honest list, not a roadmap disguised as completeness:

- **∞-category / (∞,2)-AISys** is truncated to a 1-category of interfaces and
  1-morphisms plus *data* 2-cells. There are no 3-cells, no mapping spaces,
  no homotopy-coherent associators beyond “flattening of the hom tree agrees”.
- **Univalence / real equivalence** of backends is a numerical tolerance, not
  a path in a space of functions.
- **Structured cospan pushout** identifies ports by name; it is the finite-set
  pushout, not a colimit in a topos of C-sets.
- **Transformer residuals** are representable as `ResidualAdd` components but
  the stock `transformer_block` example is the sequential backbone
  (attention → LN → FFN → LN). Skip-add wiring is in the IR op and cospan
  language; the example does not insert it automatically.
- **Training / gradients** are not implemented. Update maps on open systems
  are identity-on-weights (inference).
- **Only `f32` executes.** Other dtypes exist so the SMC can reject them.
- **No wgpu / CUDA / Metal dispatch.** WGSL is real source and is validated;
  it is not launched.
- **Cluster is simulated** (threads + channels on one machine). TCP helpers
  exist but the CLI does not spawn OS processes by default.
- **Cost model is analytical** (FLOPs / device GFLOP/s + link terms), not a
  learned performance model.
- **Sheaf values are discrete event logs**, not spaces. Compatibility is
  checksum equality on overlapping `(t, op, kind)`.

## Tests worth reading

- `catnn-core::category::smc` / `laws` — associativity and unit of composition (proptest).
- `catnn-core::lens::lawful` and `crates/catnn-core/tests/lens_laws.rs` — lens laws.
- `catnn-core::architecture::fibration` — `R(id) = id`, fuse reindexing.
- `catnn-rt::sheaf` — restrict, glue, failed descent.
- `catnn-compile` — CPU vs reference on an MLP; every generated WGSL shader parses under naga.
- `crates/catnn-rt/tests/e2e.rs` — MLP from `.catnn`, transformer, one adapt step.

## License

MIT
