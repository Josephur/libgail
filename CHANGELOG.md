# Changelog

Notable changes to GAIL. The version is the one in `include/gail/gail.h`; every release is
tagged `vX.Y.Z` and published with prebuilt archives on the
[releases page](https://github.com/Josephur/libgail/releases). Within version 1 every
release is binary compatible with the ones before it (see
[docs/LIBRARY.md](docs/LIBRARY.md#compatibility)).

## 1.0.1 — 2026-11-05

Bug-fix release, a drop-in replacement for 1.0.0 (same header, same symbols, same
codestreams for every configuration that worked before). Pull request #5.

### Library

* The rate control no longer loops forever when the per-frame budget is below the size of
  the codestream headers, and a budget under its 16-byte safety margin no longer wraps
  around and switches the limit off. `gail_encoder_create()` now rejects such a
  `max_mbps` / `fps` combination with `GAIL_ERROR_INVALID` and a message giving both sizes.
  (#1)
* The built-in DCI checker validates every marker segment length before reading it, so a
  truncated or corrupt codestream can no longer make it read past the end of the buffer. (#2)
* `gail_encoder_create()` waits until the devices have set up their frames in flight, and
  reports a failure to do so itself: `GAIL_ERROR_NOMEM` when no device has room for a
  frame, as the header has always promised, instead of `GAIL_ERROR_FAILED` from the first
  `send` / `encode` / `receive` call. (#3)

### Tools

* `gail-enc` and `gail-bench` reject non-positive `--mbps` and `--fps`. (#1)
* `gail-check` reports a file it cannot open as such, not as a malformed codestream. (#2)
* `gail-bench` prints "missing value for ..." instead of aborting when the last option has
  no value. (#4)
* `gail-enc` decodes untagged sources with one colour matrix on both input paths (BT.709
  for 720 lines or more, BT.601 below, BT.2020 when the primaries say so). Before, the
  RGB48 path (`--input-format rgb`) left the choice to FFmpeg's scaler, which takes BT.601
  at any size, so the two paths gave different colours for the same clip. The matrix is
  printed in the source line, marked as assumed when the stream carries no tag. (#6)
* A source frame that cannot be read (a short or unreadable `.xyz` file, an `ffmpeg` child
  that fails) stops the encode with an error instead of ending it silently with exit
  status 0, and `gail-enc` exits 1 when no frame was encoded at all. (#7)

## 1.0.0 — 2026-11-04

First release: DCI JPEG 2000 encoding (2K and 4K, 12-bit X'Y'Z', SMPTE ST 428-1 /
ST 429-4) with EBCOT Tier-1 on the GPU through CUDA or Vulkan, a bit-exact CPU reference
backend, multi-GPU pull scheduling, colour management for SDR and HDR sources, the C
library `libgail` with pull, callback and blocking APIs, the `gail-enc`, `gail-check` and
`gail-bench` tools, and an FFmpeg encoder (`-c:v libgail`).
