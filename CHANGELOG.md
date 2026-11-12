# Changelog

Notable changes to GAIL. The version is the one in `include/gail/gail.h`; every release is
tagged `vX.Y.Z` and published with prebuilt archives on the
[releases page](https://github.com/Josephur/libgail/releases). Within version 1 every
release is binary compatible with the ones before it (see
[docs/LIBRARY.md](docs/LIBRARY.md#compatibility)).

## Unreleased

A decoder. Everything the 1.0 header declared is unchanged, and the encoder writes the
same codestreams as 1.0.1 for the same input and settings.

### Library

* `gail_decoder`: DCI JPEG 2000 codestreams (2K and 4K, GAIL's own and other encoders')
  back to 12-bit X'Y'Z' pictures, on CUDA, Vulkan or the CPU. The same three calling
  patterns as the encoder: `gail_decoder_send()` / `gail_decoder_receive()`, the
  `on_image` callback, or `gail_decoder_decode()` from any number of threads.
* Output with the codes in the high bits of each word (`GAIL_PIX_XYZ12_MSB`, FFmpeg's
  `xyz12le`, the default) or the low bits, into the library's buffer or straight into the
  caller's with the caller's row length.
* `gail_decoder_config.reduce` leaves out resolution levels: half size, quarter size.
* `gail_probe()` reads a codestream's header and says what it is and whether it can be
  decoded here.
* Codestreams are untrusted input. A damaged or unsupported one fails on its own, with
  the reason in `gail_last_error()`, and the decoder carries on with the next.
* JPEG 2000 outside the DCI profiles (5/3 lossless, other bit depths or component counts,
  several tiles or quality layers, HTJ2K) is recognized and refused, not decoded.

### Tools

* `gail-dec` decodes a codestream or a directory of them to raw X'Y'Z' frames, times the
  decode when no output is asked for, and with `--info` prints what each codestream is.

### FFmpeg

* A `libgail` decoder next to the encoder (`integrations/ffmpeg/libgaildec.c`), added by the
  same `add-to-ffmpeg.sh`. It is selected by name (`-c:v libgail` before the input),
  decodes into FFmpeg's own frame buffers and supports `-lowres`.

### Other

* Prose, comments and messages use American English spelling throughout; the status
  text of `GAIL_ERROR_CANCELLED` is now "canceled". No name in the API changed.

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
* `gail-enc` decodes untagged sources with one color matrix on both input paths (BT.709
  for 720 lines or more, BT.601 below, BT.2020 when the primaries say so). Before, the
  RGB48 path (`--input-format rgb`) left the choice to FFmpeg's scaler, which takes BT.601
  at any size, so the two paths gave different colors for the same clip. The matrix is
  printed in the source line, marked as assumed when the stream carries no tag. (#6)
* A source frame that cannot be read (a short or unreadable `.xyz` file, an `ffmpeg` child
  that fails) stops the encode with an error instead of ending it silently with exit
  status 0, and `gail-enc` exits 1 when no frame was encoded at all. (#7)

## 1.0.0 — 2026-11-04

First release: DCI JPEG 2000 encoding (2K and 4K, 12-bit X'Y'Z', SMPTE ST 428-1 /
ST 429-4) with EBCOT Tier-1 on the GPU through CUDA or Vulkan, a bit-exact CPU reference
backend, multi-GPU pull scheduling, color management for SDR and HDR sources, the C
library `libgail` with pull, callback and blocking APIs, the `gail-enc`, `gail-check` and
`gail-bench` tools, and an FFmpeg encoder (`-c:v libgail`).
