# FFmpeg encoder and decoder: `libgail` (GAIL)

[libgailenc.c](libgailenc.c) adds a JPEG 2000 encoder named `libgail` to FFmpeg, and
[libgaildec.c](libgaildec.c) a decoder of the same name. The encoder's every packet is a
DCI-compliant codestream encoded on the GPU; the decoder turns DCI codestreams back into
`xyz12le` pictures on the GPU. Both are tested with FFmpeg 9.0.2.

## Building FFmpeg with it

1. Build and install GAIL ([docs/LIBRARY.md](../../docs/LIBRARY.md)) so that
   `pkg-config gail` works.
2. Add the encoder and decoder to FFmpeg and configure it with them enabled:

```bash
integrations/ffmpeg/add-to-ffmpeg.sh /path/to/ffmpeg
cd /path/to/ffmpeg
./configure --enable-libgail [your usual options]
make -j
```

`add-to-ffmpeg.sh` copies the two source files in and adds `--enable-libgail` to
`configure`, plus a line for each to the codec list and the Makefile. Running it twice
changes nothing, and a tree patched for the encoder alone gains the decoder.
For Windows, cross-compile FFmpeg with MinGW-w64 as usual and point `PKG_CONFIG_PATH`
at a Windows build of GAIL.

## Encoding

Encode HDR10 UHD to 4K DCI JPEG 2000 frames at 24 fps:

```bash
ffmpeg -i film.mkv -r 24 -c:v libgail -f image2 j2c/%06d.j2c
```

Scale to 2K (the picture is centered in the 1998×1080 flat container), at 25 fps:

```bash
ffmpeg -i film.mkv -r 25 -vf scale=1920:1080 -c:v libgail -f image2 j2c/%06d.j2c
```

Use FFmpeg's own R'G'B' → X'Y'Z' conversion instead of GAIL's:

```bash
ffmpeg -i film.mkv -r 24 -vf scale=1998:1080,format=xyz12le -c:v libgail j2c/%06d.j2c
```

* **Hardware:** CUDA on NVIDIA Turing or newer with driver 580+, otherwise Vulkan on any
  Vulkan 1.3 GPU, otherwise the CPU; `-backend` and `-gpus list` show what was picked.
  See [Supported hardware](../../docs/LIBRARY.md#supported-hardware).
* **Frame rate:** always set `-r` to a DCI rate (24, 25, 30, 48, 50, 60). The bitrate
  limit is per second, so the per-frame budget follows the frame rate. GAIL warns about
  other rates.
* **Wrapping:** for a DCP, wrap the `.j2c` frames with a DCP tool (`asdcp-wrap`,
  DCP-o-matic, GAIL's `scripts/make_dcp.py`). FFmpeg's own `-f mxf` is OP1a, not a DCP
  track file.

Accepted input formats:
- Y'CbCr: `yuv420p`, `yuv422p` and `yuv444p`, each at 8, 10 or 12 bits.
- Planar R'G'B': `gbrp` at 8, 10, 12 or 16 bits.
- Interleaved: `rgb48le`, and `xyz12le` (already DCI X'Y'Z').

A source already in one of these layouts goes to GAIL as it is, and GAIL converts it on
the GPU. For example, ProRes 422 stays 4:2:2, ProRes 4444 stays 4:4:4 and DPX stays planar
R'G'B'. Other layouts are converted by FFmpeg to the closest one.
Color comes from the stream's tags (transfer, primaries, matrix, range, chroma location,
and the mastering peak from HDR metadata).

| Option | Values | Default | |
|---|---|---|---|
| `-dci_profile` | `auto`, `2k`, `4k` | auto | 4K when the picture is wider than 2048 or taller than 1080 |
| `-container` | `auto`, `flat`, `scope`, `full` | auto | smallest that fits; scope for pictures wider than 2:1 |
| `-b:v` / `-mbps` | ≤ 250 Mbit/s | 230 | per-second limit; the frame limit follows from `-r` |
| `-backend` | `auto`, `cuda`, `vulkan`, `cpu` | auto | |
| `-gpus` | `list`, indices (`0,1`), UUIDs as `nvidia-smi -L` prints them, or PCI bus IDs | all | `-gpus list` prints the devices |
| `-inflight` | frames per GPU | 3 | |
| `-tonemap` | `signal`, `shoulder`, `bt2390`, `none` | signal | HDR handling ([COLOR.md](../../docs/COLOR.md)); `signal` matches DCP-o-matic |
| `-gamut` | `none`, `p3d65`, `dcip3` | none | soft gamut compression |
| `-dci_check` | 0 / 1 | 1 | verify every codestream |

## Decoding

FFmpeg keeps its own `jpeg2000` decoder as the default for JPEG 2000, so `libgail` is
chosen by name, before the input it applies to:

```bash
ffmpeg -c:v libgail -i picture.mxf -f null -              # decode a DCP picture track
ffmpeg -c:v libgail -i picture.mxf -pix_fmt rgb48le out_%06d.tiff
ffmpeg -lowres 1 -c:v libgail -i picture.mxf ...          # at half size; 2 = a quarter
```

A player that decodes through this FFmpeg's libavcodec selects it the same way: mpv with
`--vd=libgail`, VLC with `--avcodec-codec=libgail`. Neither player has been tested with it
yet.

* **What it decodes:** DCI streams, 2K and 4K: 12-bit X'Y'Z', 9/7 wavelet, one tile, as
  DCP picture track files hold them. Other JPEG 2000 (lossless, other bit depths, HTJ2K) is
  refused with a message naming the reason; the `jpeg2000` decoder handles those.
* **Output:** `xyz12le`, the format the `jpeg2000` decoder gives for the same streams, so
  whatever follows (color conversion, scaling, display) works as it did. The pictures of
  the two decoders differ by at most one code value.
* **Delay:** several frames are in flight on the GPUs at once, so a frame comes out a few
  packets after it went in, as with any decoder that has delay. Seeking works as usual.
* **Encrypted DCPs:** decryption happens where the track file is read, before the
  decoder, and is unchanged by the choice of decoder.
* **A damaged frame** is reported and skipped; decoding carries on with the next.

| Option | Values | Default | |
|---|---|---|---|
| `-lowres` | 0 to 5 | 0 | resolution levels to leave out: 1 = half size, 2 = a quarter |
| `-backend` | `auto`, `cuda`, `vulkan`, `cpu` | auto | |
| `-gpus` | `list`, `all`, indices (`0,1`), UUIDs as `nvidia-smi -L` prints them, or PCI bus IDs | the first GPU | one GPU keeps up with 4K playback several times over; `all` for the most frames per second when transcoding |
| `-inflight` | frames per GPU | 3 | |
| `-queue` | 1 to 64 | 16 | codestreams handed to the decoder before it waits for a picture |

## Tested

FFmpeg 9.0.2 on Linux. Decoding, on 2K and 4K DCPs made by GAIL and by DCP-o-matic:

| Run | Result |
|---|---|
| whole track, against the `jpeg2000` decoder | every picture within one code value; the same timestamps |
| after a seek (`-ss`) | the same |
| `-lowres 1`, `-lowres 2` | half and quarter size, within one code value of `jpeg2000 -lowres` |
| a track file cut short | the incomplete frame is reported, the rest decode |

Throughput is in [docs/BENCHMARKS.md](../../docs/BENCHMARKS.md).

Encoding, source `test-media/real_4k_hdr_30s.mkv` (HDR10):

| Run | Frames | Result |
|---|---|---|
| 2K (1920×1080 in 1998×1080) | 48 | all pass `gail-check` |
| 4K (3840×2160 in 3996×2160) | 48 | all pass `gail-check` |
| `xyz12le` | 24 | all pass `gail-check` |
| `-r 24` | — | 1.16 MB per frame, identical to `gail-enc` |
| `-f mxf` | — | decodes back as 24 JPEG 2000 frames |
