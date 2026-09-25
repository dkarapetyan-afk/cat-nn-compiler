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
cargo run -p catnn-cli -- compile mlp --cluster demo --mem code=4MiB,buffer=64GiB,scratch=128KiB
cargo run -p catnn-cli -- run mlp --cluster demo
cargo run -p catnn-cli -- train mlp --cluster demo --steps 12 --opt sgd --lr 0.05
cargo run -p catnn-cli -- train mixtral-tiny --cluster cpu --steps 4 --opt sgd --mem code=4MiB,buffer=64MiB,scratch=1MiB
cargo run -p catnn-cli -- mixtral-train-sweep --seq 2048 --global-batch 8
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
| `crates/catnn-cli` | `catnn compile \| run \| train \| adapt \| inspect \| mixtral-sweep \| mixtral-train-sweep` |

## Mapping: mathematics → Rust

Numbers refer to sections of the design document.

| Construction | Where it lives |
|---|---|
| **§1 Classical lens** `get/put` with PutGet, GetPut, PutPut | `catnn_core::lens::lawful::LawfulLens`. Laws are methods that return `Err` on failure. Property-tested on `first_of_pair` and `identity_f64`. |
| **§1 Higher lenses** | Truncated: we do not store 2-dimensional coherences of Capriotti–Danielsson–Vezzosi. The three 1-laws are the checkable fragment. |
| **§2 Indexed / dependent lenses, `R: A^op → Cat`** | `catnn_core::architecture::fibration`. `Fibre` is `R(A)`; `TotalObj` is an object of `∫R`; `reindex` is pullback along an `ArchMorphism`. `reindex_id_is_id` is the identity law of the functor. |
| **§2 Grothendieck / cartesian fibration** | Same module. Morphisms of `A` are refinements (`Id`, `Fuse`, `Lower`, `Repartition`, `TightenMemory { region, … }`). `TightenMemory` is indexed by the **region** of the memory object of `H`. |
| **§3 AISys objects = interfaces** | `catnn_core::category::Interface` — finite products of named `TensorType`s. |
| **§3 1-morphisms = open components** | `Hom` / `OpenComponent`. Sequential composition is `Tens::compose` (type-checked); parallel is `Tens::tensor`. |
| **§3 2-morphisms = refinements** | `TwoCell` + `Rewrite` (`Fuse`, `Lower`, `Reassign`, `Partition`, `Refl`, `Recompute`, `Offload`, `Shard`, `KernelResidency`, `FuseCode`) carrying an `EquivWitness`. Staging (store vs rematerialize, resident vs reload) is a 2-cell. |
| **§3 3-morphisms / ∞-coherence** | **Not implemented.** Equality of 2-cells is not a mapping space. Documented gap. |
| **§4 Open dynamical systems** | `OpenComponent { state, input, output, op, moore }`. Neural layers are Mealy (`readout` sees `U`); Moore is the flag. Sequential composition is `then` (state becomes `S⊗S'`). |
| **§4 Wiring diagrams / structured cospans** | `StructuredCospan`: `left → apex ← right`. `compose` is a port-identifying pushout of apex graphs. |
| **§5 Time site, `Run: T^op → Set`** | `catnn_rt::sheaf::{Interval, LocalSection}`. `restrict` is the restriction map. |
| **§5 Descent / gluing** | `glue` + `compatible`. Overlaps with disagreeing checksums become `DescentIssue`s on `GlobalHistory`. |
| **§6 `H` = hardware category** | `Cluster`, `Node`, `Device`, `Link`. Transfer cost is `latency + bytes/bandwidth`. |
| **§6 Memory object of `H`** | `DeviceMemory` / `MemoryHierarchy`: per-device **code**, **buffer**, **scratch** regions (capacity + bandwidth + provenance) plus host/NVMe/remote tiers. This spec **indexes** `R`. CLI `--mem code=…,buffer=…,scratch=…` or TOML `[memory.regions.*]`. |
| **Kernels as staged resources** | `KernelResource`: code size (measured or **estimate**), registers, static/dynamic SMEM, launch geometry, read/write/temp buffers. Load/evict sit on the same timeline as buffer alloc. |
| **Para(Lens) adjoints** | `catnn_core::lens::para::ParaMorphism`. `get` is the forward 1-morphism; `put` maps output cotangents to `(dP, dX)`. Sequential composition is the chain rule (Cruttwell et al.). |
| **Optimizer lens / optic** | `Sgd` is a lens `P × dP → P` (PutGet fails). `Adam` is an optic with residual `(m, v, t)`. |
| **Unified schedule** | `catnn_compile::schedule`: one timeline of kernel load/evict/launch, buffer alloc/free (first-fit arena, aliased lifetimes), transfer, recompute. Peak per region is simulated. `fastest_under(M)` and `min_region(kind)` are the per-region queries. Infeasibility **names the overflowing region**. |
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
- A cross-device tensor is routed along the cluster's directed links
  (intra-node hops use the implicit device link). Device threads run the
  ops whose inputs have arrived and read the tensor the route delivered.
  An intermediate node relays. There is no path when the topology does
  not connect the two nodes.
- `transfer::spawn_tcp_node` / `send_tcp` is a length-prefixed localhost
  protocol for a real multi-process stand-in; the default `run` path uses
  in-process hops along the same routes.
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
- **Training executes on CPU** for MLP and tiny Mixtral. Full-size Mixtral
  training is *planned* (Spark sweep + simulated timeline), not materialized.
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
- `catnn-core::spec::mixtral` — HF 8x7B / 8x22B param counts; tiny architecture type-check.
- `catnn-core::lens::moe` — expert coproduct injections; router is an unlawful optic.
- `catnn-compile::kernel::moe` — scaled-down Mixtral CPU vs independent naïve reference.
- `catnn-compile::kernel::adjoint` — f32 finite-difference smoke checks on linear / RMSNorm.
- `catnn-compile::kernel::fd64` — f64 central differences for every training adjoint (linear, softmax, CE, embedding, SwiGLU, GQA+RoPE+window, frozen-index MoE, tiny Mixtral e2e) at ~1e-6 relative.
- `catnn-compile::kernel::measure` — `nvcc -arch=sm_80` cubin / ptxas regs+SMEM; `nm --size` CPU symbols.
- `catnn-compile::schedule` — arena aliasing, peak ≤ cap, tight code → reloads, overflow names the region.
- `catnn-compile::train` — MLP/tiny Mixtral loss decreases; constrained rematerialization is bit-identical; per-region shrink.
- `catnn-compile::spark_place` — 8x22B BF16 on 1 Spark is infeasible; 8x7B FP4 fits.
- `catnn-compile::spark_train` — 8x7B train on N=1 needs NVMe offload; tiny train plan fits.

## Memory spec

The compile-time object of `H` is a **structured** budget, not a single byte count.

CLI (either flag):

```
--mem 64GiB
--mem code=4MiB,buffer=64GiB,scratch=128KiB
--mem-per-device code=4MiB,buffer=64GiB,scratch=128KiB
```

TOML (`examples/cluster.toml`):

```toml
[memory.regions.code]
kind = "code"
bytes = 16777216
bandwidth_bps = 5.0e10
latency_s = 5.0e-9
provenance = "assumption: 16 MiB resident .text"

[memory.regions.buffer]
# …

[memory.regions.scratch]
# …
```

| Region | DGX Spark / GB10 | Provenance |
|---|---|---|
| buffer | 128 GiB unified LPDDR5x @ 273 GB/s | **cited** NVIDIA DGX Spark product page |
| code | 64 MiB resident module/cubin working set | **assumption** — GB10 I-cache / cubin residency unpublished |
| scratch | 228 KiB SMEM/block | **assumption** — B200 public SMEM figure; GB10 unpublished |

Usable *buffer* on a Spark plan is 128 − 20 (OS) − 4 (workspace) = **104 GiB**. Host aliases device, so offload is NVMe (1 TiB @ 3.5 GB/s, assumed).

Queries: `fastest_under(M)` for a structured `M`; `min_region(code\|buffer\|scratch)` for the smallest cap in one dimension that still schedules. A failed schedule names the overflowing region (e.g. *code region overflow — too many distinct kernel modules resident*).

## Training

Adjoints are `Para(Lens)` 1-morphisms. The Mixtral router uses a straight-through estimator: stop-grad on the discrete top-k index, softmax weights remain differentiable. SGD is a lens (`p − lr·g`; PutGet fails). Adam is an optic with residual `(m, v, t)`.

```bash
cargo run -p catnn-cli -- train mlp --cluster demo --steps 12 --opt sgd --lr 0.05
cargo run -p catnn-cli -- train examples/mlp.catnn --cluster examples/cluster.toml --steps 8 --opt sgd
cargo run -p catnn-cli -- train mixtral-tiny --cluster cpu --steps 4 --opt sgd --mem-per-device 1MiB
cargo run -p catnn-cli -- mixtral-train-sweep --seq 2048 --global-batch 8
```

`catnn train` lowers the spec the same way `compile` does, builds the adjoint of that IR, and applies SGD or Adam. Each device is scheduled against its code, buffer, and scratch caps. The step runs only when every device fits (a tight buffer may checkpoint one activation; a tight code region may reload kernels). An overflowing region is named and nothing is executed. Cross-device activations and gradients follow the cluster links, including multi-hop relays.

The lowered Mixtral graph is the sequential chain of embedding, RMSNorm, GQA, MoE, and the LM head. The hand-written residual trainer in `train.rs` is a separate witness for the residual block. Full-size 8x7B / 8x22B weights are not materialized, so `train` refuses them. `mixtral-train-sweep` remains the Spark *plan* (joint EP/TP/PP/DP × ZeRO × checkpoint × NVMe offload, 1F1B, simulated peaks).

## Mixtral and DGX Spark

Mixtral is specified as a composite of open systems, not a hard-coded runner:

- Token embed → (RMSNorm → GQA+RoPE → RMSNorm → MoE)×L → RMSNorm → LM head.
- The MoE block is a **coproduct** of SwiGLU experts (`ExpertFamily`) plus a **router optic** (top-2-of-N softmax). Top-k is a projection, so the router is not a lawful lens (`PutGet` fails; residual = load-balance counts).
- Expert placement is a realization in `R(A)` over the Spark cluster object of `H` (EP / TP / PP).

Hyperparameters are copied from the official Hugging Face `config.json` files:

| | 8x7B (`mistralai/Mixtral-8x7B-v0.1`) | 8x22B (`mistralai/Mixtral-8x22B-v0.1`) |
|---|---|---|
| hidden / intermediate | 4096 / 14336 | 6144 / 16384 |
| layers | 32 | 56 |
| heads / KV heads | 32 / 8 | 48 / 8 |
| experts / top-k | 8 / 2 | 8 / 2 |
| vocab / ctx | 32000 / 32768 | 32000 / 65536 |
| rms_eps / rope_θ | 1e-5 / 1e6 | 1e-5 / 1e6 |
| sliding_window | `null` in HF (paper heritage 4096) | `null` |

```bash
cargo run -p catnn-cli -- mixtral-tiny
cargo run -p catnn-cli -- compile mixtral-8x7b --cluster spark4
cargo run -p catnn-cli -- mixtral-sweep
```

### DGX Spark (cited)

NVIDIA product page: GB10 Grace Blackwell, **128 GB** unified LPDDR5x, **273 GB/s**, ConnectX-7 **200 Gb/s**, up to **1 PFLOP FP4**, 20-core Arm. Compute capability **12.1 / `sm_121`** (not datacenter `sm_100`). Official Cluster Assistant: **2–4** Sparks (direct 2–3, switch 2–4). N=8 is accepted as a manual topology and flagged unsupported.

Placement uses 20 GiB OS + 4 GiB workspace reserve (usable **104 GiB** buffer). Each Spark device also has a **code** region (64 MiB resident modules, **assumption** — GB10 I-cache unpublished) and a **scratch** region (228 KiB SMEM/block from B200 public figures, **assumption** for GB10). Expert GEMMs are sharded by **EP × TP × PP**; attention is TP×PP only (replicated across EP). Search enumerates every `EP*TP*PP=N` factorization that fits and keeps the **minimum decode-step** plan (within 10%, higher EP wins). Decode time is per-node HBM at 273 GB/s plus EP all-to-all, two TP ring all-reduces per layer, and PP P2P over ConnectX-7 200 Gb/s. Training search is joint over EP/TP/PP/DP, ZeRO 0–3, checkpoint policy, NVMe offload (ZeRO-Infinity, Spark SSD **3.5 GB/s** assumed Gen4; unpublished), and bf16 vs fp8 mixed precision. CUDA kernels are emitted for `sm_121`; cubin/SASS/regs/SMEM are **measured** with `nvcc -arch=sm_80` (CUDA 12.0 in this VM; `sm_121` needs CUDA 12.9+) when `nvcc` is on PATH.

Full-size weights (~93 GB bf16 for 8x7B) are **not** materialized. Numerical check is the tiny config (same ops, seeded weights) vs an independent naïve reference. Optional `--weights` safetensors path errors clearly if the file is absent — Hugging Face transformers is **not** run in CI.

## License

MIT
