# GAIL design

## Pipeline

```
FFmpeg decode ──► pinned input buffer ──► GPU: colour ─► 9/7 DWT ─► quantise ─► Tier-1 CM ─► order ─► MQ ─► compact
 (native 4:2:0)        (per slot)          (YCbCr→X'Y'Z'   (lifting,   (dead-zone,   (warp per   (largest  (thread  (into mapped
                                            →DC shift→ICT)  6/5 levels) cblk layout)  code-block) first)    per cblk) host memory)
                                                                                                        │
CPU (per frame, in parallel): PCRD-opt rate control ─► Tier-2 packets / CPRL / POC / tile-parts / TLM ─► DCI check ─► .j2c
```

Each GPU owns several *slots* (default 3). A slot is a full set of device buffers plus a
pre-recorded Vulkan command buffer (or a CUDA stream). Worker threads, one per slot, pull
the next frame from the source, decode straight into the slot's pinned input buffer, run
the GPU part, then do rate control and Tier-2 on the CPU while the GPU already works on
other slots. A writer thread emits frames in order. Faster GPUs simply pull more frames
(e.g. 381 vs 339 frames on the 3090 vs the 5060 Ti in a mixed run).

The CPU backend runs the same stages with the reference implementations; GPU Tier-1
output is **bit-identical** to it given the same quantised coefficients (tested for every
code-block of 2K and 4K frames on both backends and both GPUs).

## DCI codestream

Fixed by the profile and checked by `gail-check` (rules taken from ISO 15444-1 Amd.1,
SMPTE 429-4 and libdcp's verifier):

| | 2K | 4K |
|---|---|---|
| Rsiz | 3 (CINEMA2K) | 4 (CINEMA4K) |
| Tiles / components / depth | 1 / 3 / 12-bit unsigned | same |
| COD | Scod=1, CPRL, 1 layer, MCT=1 (ICT), 5 levels, 32×32, no mode switches, 9/7 | 6 levels |
| Precincts | 128×128 at r=0, 256×256 elsewhere | same |
| QCD | scalar expounded, 1 guard bit | 2 guard bits |
| POC | none | one, in the main header: r0–5 then r6, CPRL |
| Tile-parts | 3 (one per component) | 6 (component × {2K part, 4K part}) |
| TLM | required (Stlm 0x50) | same |
| Size | ≤ 250 Mbit/s per frame, ≤ 200 Mbit/s per component / tile-part | same |

Only SOC, SIZ, COD, QCD, POC, COM, TLM, SOT, SOD and EOC markers are written. The COM
marker holds the text `GAIL <version>`, so a frame's encoder and build can be read from
the codestream; it counts toward the frame's size limit.

## Colour

Per pixel: Y'CbCr 4:2:0 / 4:2:2 / 4:4:4 → R'G'B' (bilinear chroma upsampling honouring
the chroma location, limited/full range, BT.601/709/2020 matrix; planar G'B'R' is just
reordered) → 16-bit index into a 64K-entry
EOTF table (BT.1886, sRGB, PQ, HLG, linear) giving cd/m² → primaries matrix to CIE XYZ
(no chromatic adaptation: D65 content stays D65) → for PQ/HLG either the DCP-o-matic-style
signal reading (default: the HDR code read as gamma 2.4) or a tone curve → optional soft
gamut compression toward neutral →
X' = 4095·(X/52.37)^(1/2.6) (SMPTE ST 428-1) → 12-bit integer → DC shift → ICT. X'Y'Z'
input passes through exactly; an app 3D LUT replaces the managed steps. One source
([src/kernels/color_math.h](../src/kernels/color_math.h)) compiles as C++, CUDA and GLSL,
with its parameters in a small float block. Details and the reasoning behind the
defaults: [COLOR.md](COLOR.md).

## Wavelet and quantisation

Irreversible 9/7 lifting (Annex F constants; low-pass ×1/K, high-pass ×K) with
whole-sample symmetric extension. Rows: one workgroup per row in shared memory. Columns:
32×64 tiles with an 8-sample halo in shared memory; the halo absorbs the ±4-sample reach of
the four lifting steps, so tiles are independent. Step sizes give every subband the same
image-domain MSE per unit coefficient error: Δb = Δbase / ‖synthesis basis‖b, encoded as
(ε, μ) and used exactly as signalled. Because power-of-two base steps produce nested
quantisers, `--step 1/2/4` decode identically at DCI rates; larger steps only save
Tier-1 work. Coefficients are stored per code-block (1024 words) as sign-magnitude with 6
extra fraction bits used for distortion estimates.

## Tier-1 on the GPU

The coder (no mode switches, single MQ codeword per code-block, as DCI decoders expect)
is split in two kernels.

**Context modelling (`t1_cm`) — one 32-lane warp/subgroup per code-block.** Lane *x* owns
column *x*. Significance, sign, "refined" and "visited" state for its 32 rows live in four
32-bit registers; neighbour columns come from subgroup shuffles, vertical neighbours are
bit shifts. For each bitplane the three passes are evaluated stripe by stripe:

* *Significance propagation* is the only pass with a causal dependency: a coefficient
  becomes eligible because an earlier coefficient *in the same pass* became significant.
  Per stripe this is solved as a monotone fixpoint over the warp (start with no new
  significance, recompute eligibility from left-column and above-in-column news,
  repeat until no lane changes — `subgroupAny`). The least fixpoint equals the sequential
  scan; typically 1–2 iterations.
* *Magnitude refinement* has no intra-pass dependency (contexts use the state after the
  significance pass).
* *Cleanup* coded sets and new significance follow directly from the masks; only the
  contexts need "state at time of coding", which is a fixed combination of masks (left
  column incl. its new bits, right column without, above-in-column with, below without).
  Run-length mode checks a 3×6 neighbourhood window.

Every lane emits its column's symbols (≤10 per stripe and pass, packed 6 bits each into a
64-bit register); a subgroup prefix sum puts them in scan order in shared memory, and each
bitplane's symbols are copied to a global chunk. Distortion reductions are exact integers
(no floating point), so all backends agree bit for bit.

**MQ coding (`t1_mq`) — one thread per code-block.** Code-blocks are first ordered by
symbol count, largest first (a one-workgroup counting sort), which shortens the critical
path and keeps similar-length blocks in the same warp (lane efficiency 64–72% → 99%).
The coder is written branch-free: with a = A − Qe and exch = (a < Qe), both CODEMPS and
CODELPS reduce to "C += Qe iff isMPS ≠ exch, else A = Qe"; the context moves to
NMPS/NLPS only when renormalisation follows. Contexts hold their probability-table entry
(plus MPS bit) in shared memory so the coder state stays in registers. These two changes
cut the kernel by ~20%.

**Exact truncation lengths.** For every coding pass the coder state (C, A, CT, pending
byte, position) is snapshotted; after the final flush each pass gets the *smallest*
codeword prefix that a standard decoder (which feeds 1-bits past the end of a segment)
decodes identically: the prefix plus its all-ones tail must lie in the coding interval
[low, low + A) at that pass, evaluated in 64-bit fixed point with bit-stuffing-aware byte
weights; prefixes never end in 0xFF. Verified by decoding every truncation point of
thousands of random code-blocks with an independent decoder. (An earlier version kept
the propagated carry as a 9th bit of the pending byte; fixing it shortened truncation
points and gained 2.3 dB at the same rate.)

**Early termination (rate-distortion skipping).** At DCI rates most fine bitplanes are
discarded by rate control. With threshold τ = 0.25 × λ of the latest frame, a code-block
stops after a bitplane when both the plane's estimated slope (ΔD / estimated bytes) and
the *entire remaining* error divided by the same bytes are below τ. Bytes are estimated
from the plane's empirical entropy (per-context symbol counts — run-length symbols are
nearly free), which avoids stopping sparse blocks early. If a frame's final λ comes out
below τ (an easier scene), the frame is re-coded without skipping. This halves the
symbols (90 → 47 M per 4K frame) with identical decoded PSNR.

## Rate control and Tier-2 (CPU)

Per code-block convex hull of (rate, distortion) over passes; distortion is weighted by the
subband synthesis energy and the inverse-ICT component gain. The global slope λ is found by
bisection on a fast size model: data bytes as suffix sums of a 1/1024-octave histogram of
hull slopes, packet headers as a calibrated linear function of included blocks. A few
exact Tier-2 size passes recalibrate the model; the final size is always measured
exactly and nudged until it fits. A second search per component enforces the 200 Mbit/s
component / tile-part cap. Tier-2 writes tag trees, Lblock, pass counts and the CPRL
progression (with the 4K POC) exactly as Annex B prescribes, splits tile-parts by
component (and by POC for 4K) and back-patches TLM and Psot.

## Multi-GPU and concurrency notes

* CUDA: one stream per slot; frames overlap on the GPU, hiding the serial tail of the MQ
  kernel. Slot threads block (not spin) in `cudaStreamSynchronize`, leaving the cores to
  the rate control of the other slots. Results are compacted into device memory and
  fetched with three exact-size copies rather than written by the GPU into mapped host
  memory (far fewer PCIe transactions).
* Vulkan: uploads run on a transfer-only queue (DMA engines) overlapped with compute;
  results are written by the GPU straight into host-visible memory. Each slot submits to
  its own compute queue (up to 8, `GAIL_VK_QUEUES` overrides): with 4 frames in flight
  that lifts an RTX 5060 Ti from 75 to ~95 fps at 4K and 132 to ~230 fps at 2K, and an
  RTX 3090 from 134 to ~190 fps at 2K; on the 3090 at 4K frames contend instead (101 →
  ~80 fps), where CUDA is the backend to use anyway. Event-based intra-frame sync
  (`GAIL_VK_EVENTS`) did not help on NVIDIA.
* Tier-1: the context-modelling kernel quantises its code-block straight from the wavelet
  planes (no coefficient round trip through memory; the coefficient buffer is only filled
  for the debug taps and the CPU fallback). The MQ kernel codes 16 symbols per fully
  unrolled chunk and prefetches the next symbol's context state, keeping the shared-memory
  load out of the per-symbol dependency chain (MQ 3.9 → 3.1 ms on the 5060 Ti, 5.6 → 4.1 ms
  on the 3090 at 4K).
* Device memory: a slot owns ~0.7 GB (4K) / ~0.2 GB (2K) of buffers, and the pipeline
  treats the GPU as shared. Each backend reports the exact per-slot footprint (the buffer
  sizes it is about to allocate, rounded to 2 MiB pages; measured 3–13 % above what the
  driver charges) and the live budget (`cudaMemGetInfo`; `VK_EXT_memory_budget`, heap
  budget minus usage). `fitSlots()` starts as many frames in flight as fit under the budget
  less a headroom of 512 MiB or 6 % of the card, allocation failures are caught per slot
  (`DeviceOutOfMemory`, with nothing leaked: slot construction is exception-safe) and the
  run continues with the slots that exist, and every half second one slot per device re-reads
  the budget and retires itself (own-queue drain only, other slots keep going) when less
  than half the headroom is left, so a process that starts taking memory later gets it.
  Measured on an RTX 3090 with a CUDA process growing by 1 GB/s: without this, the other
  process hits out-of-memory while the encoder keeps its 4 slots; with it, the encoder
  steps down to one slot and the other process gets ~2 GB more. On Linux the NVIDIA driver
  never evicts the encoder's own allocations (the *other* process fails), so this is about
  being a good neighbour; on Windows, where WDDM shrinks budgets and pages, it also keeps
  the encoder itself off the slow path.
* Capacity overflows (pathological content exceeding the symbol / codeword buffers) are
  detected on the GPU and the frame's Tier-1 is redone on the CPU from the GPU's
  coefficients, so output stays exact.
