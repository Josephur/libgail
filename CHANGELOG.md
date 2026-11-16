# Changelog

Notable changes to GAIL. The version is the one in `include/gail/gail.h`; every release is
tagged `vX.Y.Z` and published with prebuilt archives on the
[releases page](https://github.com/Josephur/libgail/releases). Within version 1 every
release is binary compatible with the ones before it (see
[docs/LIBRARY.md](docs/LIBRARY.md#compatibility)).

## Unreleased

Fixes to the decoder. The header and the binary interface are unchanged.

### Library

* `gail_decoder_cancel()` while a `gail_decoder_decode()` had its frame on a device let
  that call return `GAIL_OK` with the picture. It now waits for the device to finish
  with the frame, as before, and returns `GAIL_ERROR_CANCELLED` without the picture.
* A first codestream the decoder could not be set up from (damaged or unsupported) had
  its image passed to `on_image` inside `gail_decoder_send()`, on the app's thread. Such
  images now come from a decoder thread, in their place in the order, like all others.
* A device failure in the middle of the stream (a lost GPU, a driver error) failed only
  the frame it happened on, and later codestreams kept going to that device. It now
  stops the decoder, as a failure while setting up does: `gail_decoder_error()` says
  what happened, the frames queued before it fail with `GAIL_ERROR_FAILED` without going
  to a device, and later sends and decodes return the failure. A damaged or unsupported
  codestream still fails only its own frame.
* `gail_decoder_send()` from inside `on_image` could hang with one frame in flight and
  another codestream queued, and so could `gail_decoder_decode()` from inside
  `on_image`. A send from the callback now never waits, and a decode from it runs on the
  callback's own thread, on the device of the frame being delivered, which is idle
  meanwhile.

## 1.1.0 — 2026-11-15

A decoder. Everything the 1.0 header declared is unchanged. The encoder codes frames as
1.0.1 does, except that the COM marker now names the kind of device that encoded each one.

### Library

* `gail_decoder`: DCI JPEG 2000 codestreams (2K and 4K, GAIL's own and other encoders')
  back to 12-bit X'Y'Z' pictures, on CUDA, Vulkan or the CPU. The same three calling
  patterns as the encoder: `gail_decoder_send()` / `gail_decoder_receive()`, the
  `on_image` callback, or `gail_decoder_decode()` from any number of threads.
* Output with the codes in the high bits of each word (`GAIL_PIX_XYZ12_MSB`, FFmpeg's
  `xyz12le`, the default) or the low bits, into the library's buffer or straight into the
  caller's with the caller's row length.
* `gail_decoder_config.reduce` leaves out resolution levels: half size, quarter size.
* A decoder with no devices named uses the first device its backend lists, since one GPU
  decodes 4K several times faster than it plays; `num_devices = GAIL_ALL_DEVICES` with
  `devices` NULL uses every device.
* `gail_probe()` reads a codestream's header and says what it is and whether it can be
  decoded here.
* Codestreams are untrusted input. A damaged or unsupported one fails on its own, with
  the reason in `gail_last_error()`, and the decoder carries on with the next. A device
  failure stops the decoder for good: later calls return it, and the decoder is not set
  up again from another codestream.
* The decoder holds only a few decoded pictures the app has not taken yet, so a slow
  consumer does not make memory grow: with `on_image`, `gail_decoder_send()` waits for
  the callback; in pull mode it returns `GAIL_EAGAIN`, taking nothing, until a picture
  has been received.
* A destination buffer that gives its `width` and `height` only takes a picture of that
  size.
* The COM marker of every frame names the kind of device that encoded it after the version:
  `vulk` for Vulkan, `cuda` and the toolkit version for CUDA (`GAIL 1.1.0 cuda13.4`),
  nothing for the CPU. The frame budget allows for the longest tag, which leaves 10 bytes
  fewer for coded data than 1.0.1 did, so a frame at the bitrate cap can occasionally
  come out slightly different. See [docs/LIBRARY.md](docs/LIBRARY.md#versions).
* JPEG 2000 outside the DCI profiles (5/3 lossless, other bit depths or component counts,
  several tiles or quality layers, HTJ2K) is recognized and refused, not decoded.

### CUDA backend

* The CUDA backend supports Maxwell, Pascal and Volta GPUs (compute capability 5.0 to
  7.0) as well as Turing and newer. It is now two plugins next to the gail library, built
  from the same sources: `gail_cuda12` (CUDA 12.9) for Maxwell, Pascal and Volta, and
  `gail_cuda13` (CUDA 13.4) for Turing and newer. The shared library itself links no CUDA.
  Applications ship both files in the library's directory; nothing in the API changes.
  See [docs/LIBRARY.md](docs/LIBRARY.md#cuda-plugins).
* `GAIL_BACKEND_AUTO` no longer picks CUDA for a GPU the CUDA backend has no code for, and
  goes on to Vulkan when CUDA can run none of the GPUs.
* Listing devices goes by each GPU's architecture and sets nothing up on it: no CUDA
  context and no device memory stays behind. The kernels are loaded when an encoder or
  decoder sets a GPU up, and a GPU they do not load on is left out then, with the reason:
  for good when the plugin has no code it runs, for that encoder or decoder only when
  they cannot load now (its memory full, the GPU busy or in exclusive use), so that a
  later one tries it again.
* With `GAIL_BACKEND_CUDA`, a GPU that has no code in either plugin, or that its kernels
  do not load on, is left out like that too; creating the encoder or decoder fails, with
  the reason, only when that GPU was named in `devices` / `device_ids` or no CUDA GPU is
  left.
* `gail-enc --list-devices` shows the plugin that serves each CUDA device, and
  `gail_version_string()` names the plugins loaded.
* Build: `GAIL_CUDA13_ROOT` and `GAIL_CUDA12_ROOT` name the two toolkits, and
  `GAIL_CUDA13_ARCHS` / `GAIL_CUDA12_ARCHS` replace `GAIL_CUDA_ARCHS`. The static library
  links one backend in, as before.

### Tools

* `gail-dec` decodes a codestream or a directory of them to raw X'Y'Z' frames, times the
  decode when no output is asked for, and with `--info` prints what each codestream is.
  `--gpus` takes indices or device IDs (UUIDs, PCI bus IDs) as `gail-enc` does, and an
  ID that matches no device is an error. Installed, it finds the shared library in the
  `lib/` next to its `bin/`.

### FFmpeg

* A `libgail` decoder next to the encoder (`integrations/ffmpeg/libgaildec.c`), added by the
  same `add-to-ffmpeg.sh`. It is selected by name (`-c:v libgail` before the input),
  decodes into FFmpeg's own frame buffers and supports `-lowres`. A codestream of another
  size than the stream's first is refused rather than decoded into a frame of the wrong
  size. FFmpeg's `configure` now looks for a libgail that has the decoder.

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
