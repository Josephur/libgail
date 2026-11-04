# Colour handling

GAIL turns source pictures into 12-bit DCI X'Y'Z' (SMPTE ST 428-1) before the wavelet
transform. Everything here runs on the GPU in the colour kernel, from one shared source,
[src/kernels/color_math.h](../src/kernels/color_math.h). It is compiled as C++ (CPU
reference), CUDA and GLSL, and `gail-tests` checks all three against each other in every
mode. Results agree to within one code value.

There are three modes. Apps choose one through the API ([include/gail/gail.h](../include/gail/gail.h)),
and `gail-enc` exposes the same options as flags.

| Mode | Use when | API | `gail-enc` |
|---|---|---|---|
| Managed | the source is video (SDR or HDR) and GAIL should do the conversion | `GAIL_COLOR_MANAGED` + a preset | default |
| X'Y'Z' pass-through | the app already converted to X'Y'Z' (e.g. DCP-o-matic's own colour pipeline, a DI deliverable) | `input.format = GAIL_PIX_XYZ12` / `_MSB` | `--xyz-dir DIR` |
| 3D LUT | the app or colourist has a LUT from source R'G'B' to X'Y'Z' | `GAIL_COLOR_LUT3D` (in memory) | `--lut3d FILE.cube` |

Pass-through is exact: every 12-bit code survives unchanged (tested for all 4096 codes).

## Managed pipeline

```
R'G'B' (or Y'CbCr 4:2:0 / 4:2:2 / 4:4:4, or planar G'B'R' -> R'G'B' on the GPU)
  -> EOTF (BT.1886 / sRGB / PQ / HLG / linear / gamma 2.6), cd/m2
  -> primaries matrix to CIE XYZ (no chromatic adaptation: D65 content stays D65)
  -> tone map (PQ / HLG only), applied in the projector gamut
  -> soft gamut compression into the projector gamut (P3-D65 by default)
  -> X' = 4095 * (X / 52.37)^(1/2.6)
```

SDR sources map code value 1.0 to `sdr_white` (48 cd/m², cinema peak white) and have no
tone map.

### HDR sources

By default, PQ and HLG are handled the way DCP-o-matic handles them. DCP-o-matic does not
tone-map: libdcp's RGB → XYZ conversion reads the HDR signal as if it were gamma 2.4 video,
with code 1.0 at 48 cd/m² (`--tonemap signal`, `GAIL_TONE_MAP_SIGNAL`). The result is
dimmer and flatter than a true HDR rendering, but no highlight is crushed and no colour
glows, so it is a safe default for apps that pass no colour settings of their own.

Apps that want an actual HDR rendering can pick a tone curve:

* **`shoulder`** (scene-relative):
  * HDR reference white (`hdr_ref_white`, 203 cd/m², ITU-R BT.2408) is scaled to
    `cinema_ref_white` (16 cd/m²).
  * Everything below `knee` × peak (0.25 × 48 = 12 cd/m²) is scaled linearly.
  * Above the knee, an extended-Reinhard shoulder reaches the cinema peak at the source's
    mastering peak (`hdr_peak`, from the file's metadata in `gail-enc`).
* **`bt2390`:** the ITU-R BT.2390 EETF on absolute PQ luminance (bright).
* **`none`:** absolute luminance; anything above the cinema peak clips.

### What the curve is applied to (shoulder / BT.2390)

| `tone_norm` / `--tone-norm` | Bright saturated colours | Notes |
|---|---|---|
| `channel` (default) | drift toward white as each channel nears the peak | how film and SDR grades render them; keeps texture in highlights |
| `maxrgb` | keep hue and saturation exactly | bright warm light stays deep orange; can look over-saturated |
| `luminance` | keep luminance, so the gamut stage bleaches them to white | washes out detail in bright, saturated scenes |

Neutrals are identical in all three modes. Out-of-gamut (negative) channels are scaled
with the brightest channel, so an out-of-gamut colour shrinks in proportion instead of
being pulled toward black.

### Gamut

By default there is no gamut stage, as in DCP-o-matic: X'Y'Z' can hold every source
colour, and the projector maps what it cannot show. With `--gamut p3d65` or `dcip3`,
colours outside that gamut are not clipped per channel. They are moved toward the neutral of the same luminance, and only as
far as needed. Compression starts at `gamut_threshold` (0.8) of the way to the gamut edge
and rolls off smoothly, so texture inside saturated areas survives.

## Choosing the defaults

Defaults matter for apps that pass no colour information, so they follow the established
tool. Test clip: 4K PQ BT.2020 HDR10, mastering peak 2000 cd/m², compared frame by frame
with DCP-o-matic's 2K DCP of the same clip.

| Default tried | Result on bright stage-light frames (431, 471, 575) | X'Y'Z' PSNR vs DCP-o-matic |
|---|---|---|
| BT.2390 EETF | very bright, highlights crushed | — |
| shoulder 203→36, luminance | 39–64 % of pixels within 8 cd/m² of peak; saturated light bleached | 21–26 dB |
| shoulder 203→16, per channel | closer to libplacebo's BT.2446a rendering; still visibly more saturated and glowing than DCP-o-matic | 21–26 dB |
| **signal (current)** | matches DCP-o-matic | 45–49 dB on single frames, limited by DCP-o-matic's own lossy encode and frame matching |

The remaining difference from DCP-o-matic comes from small rounding differences in the
conversion. The tone-curve modes stay available for apps that want a true HDR rendering.

## Using it from an app

```c
gail_config cfg;
gail_config_defaults(&cfg);
cfg.profile = GAIL_PROFILE_4K;
cfg.input.format = GAIL_PIX_YUV420P10;       /* or GAIL_PIX_RGB48 */
cfg.input.width = 3840;                      /* centred in the 3996x2160 container */
cfg.input.height = 2160;
cfg.input.matrix = GAIL_MATRIX_BT2020;
gail_color_preset(GAIL_PRESET_HDR_PQ_BT2020, &cfg.color);  /* DCP-o-matic-like */
/* optional, a true HDR rendering instead: */
cfg.color.tone_map = GAIL_TONE_MAP_SHOULDER;
cfg.color.hdr_peak = mastering_peak;         /* from the source's metadata */
cfg.color.gamut = GAIL_GAMUT_P3D65;
gail_encoder* enc;
if (gail_encoder_create(&cfg, &enc) != GAIL_OK) { puts(gail_last_error()); return; }
```

An app with its own colour pipeline (DCP-o-matic converts to X'Y'Z' itself) sets
`cfg.input.format = GAIL_PIX_XYZ12` (codes in the low bits, as libdcp produces them) or
`GAIL_PIX_XYZ12_MSB` (FFmpeg's `xyz12le`). The codes are then encoded exactly and
`cfg.color` is not used. `gail_color_describe()` returns the same one-line summary
`gail-enc` prints, which is useful for logs.

Building, linking and the rest of the API: [LIBRARY.md](LIBRARY.md).
