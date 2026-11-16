# Benchmarks

Machine: RTX 5060 Ti 16 GB (sm_120) + RTX 3090 24 GB (sm_86), AMD Ryzen 9 5900X (12 cores, 24 threads), NVIDIA driver
615.78, CUDA 13.4, Vulkan 1.4. Content: 48 frames from 10 s into the test clip (4K HDR
PQ), preloaded as native Y'CbCr 4:2:0, looped 6× (288 frames), 230 Mbit/s at 24 fps,
early termination 0.25, 4 frames in flight per GPU, DCI check on every frame.

```bash
./build/gail-bench --backend cuda --profile 4k -i test-media/real_4k_hdr_30s.mkv --start 10 --frames 48 --loops 6 --rd-skip 0.25 --slots 4 --check
```

## Throughput (frames/s)

| Profile | Backend | RTX 3090 | RTX 5060 Ti | both |
|---|---|---:|---:|---:|
| 4K 3996×2160 | CUDA   | 171.1 | 119.5 | 246.7 |
| 4K 3996×2160 | Vulkan |  75.7 |  87.0 | 152.2 |
| 2K 1998×1080 | CUDA   | 328.5 | 298.5 | 551.7 |
| 2K 1998×1080 | Vulkan | 196.9 | 200.4 | 365.6 |
| 4K           | CPU reference, 24 threads | | | 8.59 |
| 2K           | CPU reference, 24 threads | | | 24.8 |

Each figure is the best of four runs taken in two interleaved passes over the whole
matrix on an otherwise idle machine (throughput here drifts by 10–20 % between sessions and
with anything else running, so single runs are not quoted). Every frame of every run
passed the DCI check.

End to end from the H.264 file with `gail-enc` (FFmpeg decode included, both GPUs, auto
backend = CUDA): 720 frames 4K in 5.67 s = **127 fps**; 2K 720 frames in 4.91 s = 147 fps. FFmpeg's H.264
decode (≈150 fps output-equivalent for this 59.94 → 24 fps clip) is then the limit.

## Per-frame GPU stage times, 4K, Vulkan, one frame in flight (RTX 3090)

| stage | ms |
|---|---:|
| color (Y'CbCr → X'Y'Z' → ICT) | 0.17 |
| 9/7 DWT, 3 components, 6 levels | 0.71 |
| quantization | in Tier-1 |
| Tier-1 context modeling (incl. quantization) | 3.05 |
| Tier-1 MQ coding | 4.0 |
| compaction + readback | 0.7 |

Tier-1 is ~80% of GPU time. Rate control + Tier-2 on the CPU take ~3–8 ms per frame on
one thread and run in parallel with GPU work.

## Effect of the main optimizations (4K, RTX 3090, Vulkan)

| change | effect |
|---|---|
| tiled shared-memory vertical DWT | DWT 4.4 → 0.75 ms |
| upload on transfer queue, compact readback, decode into pinned memory | 42 → 63 fps |
| entropy-based early termination | Tier-1 symbols 90.6 M → 47 M per frame, same PSNR |
| MQ contexts in shared memory, largest-first ordering | MQ 7.2 → 5.8 ms |
| branch-free MQ coder | MQ 6.6 → 5.5 ms |
| model-based rate control | CPU rate control ~25 → ~5–13 ms per frame |
| native Y'CbCr input (GPU color conversion) | 4× less upload; end to end 23–55 → 113 fps |
| exact truncation lengths (carry fix) | +2.3 dB at the same bitrate |
| MQ: 16-symbol unrolled chunks, context-state prefetch | MQ 5.6 → 4.1 ms (CUDA, 3090) |
| quantization fused into context modeling; compaction via device staging + exact copies (CUDA) | GPU time per 4K frame 12.6 → 11.2 ms (5060 Ti) |
| Vulkan: one compute queue per slot | 2K 134 → ~190 fps (3090), 132 → ~230 fps (5060 Ti) |
| CPU backend: strip-wise DWT, thread pool, packed Tier-1 flags, two slots | 2K 6.9 → 24 fps, 4K 2.0 → 8.4 fps |

## Sharing the GPU

Frames in flight are sized to the device memory that is free, and given up during a run when
another process needs the memory (see [DESIGN.md](DESIGN.md)). Measured on the RTX 3090 with
an artificial CUDA process (4K, 4 frames in flight requested):

| scenario | before | now |
|---|---|---|
| other process holds 22 GB of 24 (1.6 GB free) | `VK_ERROR_OUT_OF_DEVICE_MEMORY` / `cudaMalloc out of memory`, encode aborts | 1 of 4 frames in flight, encode runs |
| other process holds 23.3 GB (0.5 GB free), explicit backend | abort | clear error ("no device has memory for a frame …") |
| same, `--backend auto` | abort | encodes on the CPU |
| other process grows by 1 GB/s while encoding (Vulkan) | encoder keeps 4 frames; other process fails at 21.0 GB | encoder steps down to 1 frame; other process reaches 22.0 GB |
| same (CUDA) | other process fails at 20.0 GB | encoder steps down to 2 frames; other process reaches 21.0 GB |

On Linux the NVIDIA driver never evicts the encoder's allocations: the process that
allocates last is the one that fails, so this is about leaving room for others (and about
the encoder itself on Windows, where the budget shrinks and memory is paged).

Compute contention (a CUDA kernel loop on the 3090 at 100 % / 50 % duty, 4K, bench defaults,
single GPU): CUDA 310 → 169 / 230 fps; Vulkan with 1 queue 194 → 84 / 136 fps, with 8
queues 180 → 60 / 113 fps. Nothing in the encoder changes the driver's time-slicing
between processes; the queue count only matters for the 3090 at 4K, where one queue is
better with or without a neighbor (`GAIL_VK_QUEUES=1`).

## Things that did not help (NVIDIA)

* One Vulkan compute queue per frame slot on the RTX 3090 at 4K: 101 → ~80 fps (frames
  contend; it is a large win at 2K and on the RTX 5060 Ti, so it is the default).
* Vulkan events instead of pipeline barriers to let frames overlap on one queue: no change.
* Removing the integer modulo from the DWT lifting loops: no change (memory-bound).
* FP16 / BF16 / TF32 DWT (including a 3×TF32 tensor-core matrix form): FP32 is already
  memory-bound (≈0.3 ms per 4K plane), so none of them is faster. FP16 and BF16 lose far too
  much (quantized coefficients differ in 58 % / 73 % of the finest-band positions); plain
  TF32 still differs in 22 %, and the 3×TF32 split that recovers FP32-class accuracy costs
  4× the time.

## Decoding

Machine as above, NVIDIA driver 620.70. Content: 48 frames from 15 s into a 30 s DCP made
by DCP-o-matic at 250 Mbit/s and 24 fps (the frames average 245 Mbit/s; a DCP of the same
clip made by GAIL decodes at the same rates), preloaded and looped 10×, 3 frames in flight
per GPU, pictures returned in host memory as `xyz12le`. The first frame, which sets the
devices up, is decoded before the clock starts.

```bash
./build/gail-dec -i frames/ --loops 10 --backend cuda
```

| Profile | Backend | RTX 3090 | RTX 5060 Ti | both |
|---|---|---:|---:|---:|
| 4K 3996×2160 | CUDA   | 121.4 | 57.3 | 160.0 |
| 4K 3996×2160 | Vulkan | 100.7 | 51.9 | 148.7 |
| 2K 1998×1080 | CUDA   | 189.5 | 92.2 | 252.7 |
| 2K 1998×1080 | Vulkan | 155.2 | 84.9 | 236.9 |
| 4K at half size (`--reduce 1`) | CUDA | | | 301.4 |
| 2K at half size (`--reduce 1`) | CUDA | | | 566.6 |
| 4K           | CPU, 24 threads | | | 15 to 22 |
| 2K           | CPU, 24 threads | | | 29 to 43 |

Each GPU figure is the median of three runs. Other work was running on the machine during
all of them, taking between a third and all of its cores, so the figures are on the low
side; the CPU rows are given as the range seen across sessions for that reason.

For scale, FFmpeg 9.0.2's own `jpeg2000` decoder (16 frame threads) on the same track
files and the same machine decodes 13.4 to 17.6 fps at 4K and 30.7 to 44 fps at 2K, the
spread again being what else the machine was doing. Run back to back with GAIL's CPU
backend under the same load, it gave 13.4 fps at 4K against GAIL's 17.9, and 32 fps at
2K against 34. OpenJPEG 2.5.4 decodes a 4K frame in about 340 ms on one thread.

Through FFmpeg with the `libgail` decoder (`ffmpeg -gpus all -c:v libgail -i picture.mxf -f null -`,
both GPUs, the whole 724-frame track including the MXF demuxer): 121 to 137 fps at 4K and
273 to 289 fps at 2K, against 16.5 and 43 fps with `-c:v jpeg2000`. CPU time for the 4K
track: 14 s with `libgail`; the `jpeg2000` decoder used 177 s for a third of it.

### Per-frame stage times, 4K, CUDA, one frame in flight

| Stage | RTX 3090 | RTX 5060 Ti |
|---|---:|---:|
| upload, Tier-1 and dequantization | 4.0 ms | 9.3 ms |
| inverse wavelet (6 levels, 3 planes) | 0.8 ms | 1.4 ms |
| inverse ICT, packing and download | 4.0 ms | 8.4 ms |
| whole frame, wall clock, with header and packet parsing on the host | 14.7 ms | 25.2 ms |

These are for a test-pattern frame at the 250 Mbit/s limit.

### What made the difference

| Change | Effect |
|---|---|
| Code-blocks packed into warps by a byte budget (long blocks alone, short ones together) in place of 32 blocks per warp sorted by length | Tier-1 for a 4K frame 11.9 → 3.8 ms on the RTX 3090, 10.5 → 6.5 ms on the RTX 5060 Ti |
| Picture buffers reused in place of a fresh 52 MB allocation per 4K picture | 4K on the RTX 3090 with 3 frames in flight: 61 → 121 fps |
| More frames in flight (the default is 3 per GPU) | RTX 3090 at 4K: 1 / 3 / 6 frames give 46 / 116 / 149 fps |
