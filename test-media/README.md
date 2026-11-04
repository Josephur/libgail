# Test media

This directory is not in the repository: the clips are large and not ours to
redistribute. The tests and benchmarks that need real video look here and skip
themselves when it is empty.

* `real_4k_hdr_30s.mkv`: a 30-second 3840×2160 HDR10 clip (H.264, BT.2020 PQ,
  mastering peak 2000 cd/m²). It is used by the Y'CbCr tests in `gail-tests`, the
  benchmarks in [docs/BENCHMARKS.md](../docs/BENCHMARKS.md), the validation runs in
  [docs/VALIDATION.md](../docs/VALIDATION.md) and the FFmpeg integration's tests. Any HDR10
  UHD clip with those characteristics serves the same purpose; cut 30 s of it with

  ```bash
  ffmpeg -ss 0 -t 30 -i your_hdr10_clip.mkv -c copy test-media/real_4k_hdr_30s.mkv
  ```
