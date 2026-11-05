# Validation

DCI cinema compatibility is checked at six levels, run on every directory of encoded
frames before a release. Only the first level ships in this repository (`gail-check`); the
others are external tools and the harness that drives them is not part of the library.

1. **Own DCI checker** (`gail-check`, also run on every frame inside `gail-enc`): parses
   every marker and enforces the cinema profile rules from ISO 15444-1 Amd.1 /
   SMPTE ST 429-4 and the libdcp/ClairMeta verifier: Rsiz, single tile, 3 × 12-bit
   unsigned components, COD (CPRL, 1 layer, MCT, 5/6 levels, 32×32, no mode switches,
   9/7, precincts 128/256), QCD (scalar expounded, 1/2 guard bits, ≤ 30 magnitude
   bitplanes), the exact 4K POC in the main header, 3/6 tile-parts in order, TLM present
   and equal to Psot, only allowed markers, no marker-like `0xFF 0x90+` pairs inside packet
   data, EOC, frame ≤ 250 Mbit/s (warning above 230) and component / tile-part ≤ 200 Mbit/s.
2. **jpylyzer 2.2.1** on every frame: codestream validity, and every reported DCI property
   cross-checked against the profile.
3. **Two independent decoders**, OpenJPEG 2.5.4 (`opj_decompress`) and FFmpeg's native
   JPEG 2000 decoder, on sampled frames; PSNR against the encoder's own 12-bit X'Y'Z'
   reference (`gail-enc --dump-xyz N`) and agreement between the decoders.
4. **SMPTE MXF wrapping** with asdcplib's `asdcp-wrap -L` (which refuses frames with
   inconsistent parameters), plus a minimal SMPTE DCP (CPL with markers and composition
   metadata, PKL, ASSETMAP, VOLINDEX) from `scripts/make_dcp.py`.
5. **libdcp `dcpverify`** on that DCP, including its JPEG 2000 bitstream checks
   (`verify_j2k`) and per-frame bitrate check. A negative test (one frame patched to 1
   guard bit) confirmed it reports "The JPEG2000 codestream uses 1 guard bits in a 4K
   image instead of 2".
6. **DCP-o-matic's verifier.** `dcpomatic2_verifier` is a GUI; a small headless harness
   reproduces its Verify button: the same `dcp::verify()` call from the system libdcp
   (1.10.63, the one DCP-o-matic 2.18.50 links), after `dcp::init()` with DCP-o-matic's
   libdcp resource directory, picture-detail checks on, and the same Errors / SMPTE Bv2.1
   errors / Warnings classification. DCP-o-matic only marks a job
   failed on Errors, but some codestream violations (e.g. wrong guard bits) are reported
   as Bv2.1 errors, so the check requires zero notes in all three categories. The
   corrupted-guard-bits DCP is reported here too.

Unit tests (`gail-tests`): DWT round trip; Tier-1 encoder vs an independent Tier-1
decoder at **every** truncation point of 400 random code-blocks (11,494 points) and of
pass-limited blocks (4,683 points); full CPU frames through the DCI checker; GPU Tier-1
(Vulkan and CUDA, both GPUs) **bit-identical** to the CPU reference for every code-block
of 2K and 4K frames; GPU DWT vs CPU (max difference 0.003); GPU Y'CbCr (4:2:0, 4:2:2,
4:4:4) and planar R'G'B' conversion vs CPU reference (≤ 1 code value); pipeline failure
injection (a device dying mid-frame, refused frame slots, a failing source or sink, a
cancel from another thread). The API test (`gail-api-test`) covers every calling pattern,
every input format, cancelling and the binary-compatibility rules.

## Results: full 30 s test clip (720 frames, 24 fps, 230 Mbit/s target)

Source: `test-media/real_4k_hdr_30s.mkv`, 3840×2160 H.264, BT.2020 PQ, 2000 cd/m².

| | 4K 3996×2160 | 2K 1998×1080 |
|---|---|---|
| gail-check | 720 / 720 compliant | 720 / 720 compliant |
| jpylyzer | 720 / 720 valid, all properties as required | 720 / 720 valid, all properties as required |
| Bitrate | max 1,197,900 B (230.0 Mbit/s), mean 218.7 Mbit/s | max 1,197,900 B, mean 218.0 Mbit/s |
| Decoders | 90 frames, both decoders OK; PSNR mean 55.90 dB, min 41.62 dB; decoders agree to 90.2 dB | 90 frames; PSNR mean 65.32 dB, min 46.98 dB; agree to 90.2 dB |
| MXF + dcpverify | no notes; "each frame … safely under the limit of 250Mbit/s" | same |
| DCP-o-matic verifier | 0 errors, 0 Bv2.1 errors, 0 warnings | 0 errors, 0 Bv2.1 errors, 0 warnings |

## Perceptual quality review

Every frame that has an X'Y'Z' reference (`gail-enc --dump-xyz N`) is decoded with
OpenJPEG and compared with the encoder's input, so only the encoder's loss is measured,
not the HDR → cinema colour conversion. Metrics: PSNR per component,
and on the Y' (luminance) channel SSIM, XPSNR and VMAF (4K model at 4K). VMAF is scored on
isolated stills, where its motion feature is zero, so even a perfect frame cannot reach
100 (about 97.4 with the 1080p model); each reference is also scored against itself and
the *loss* from that ceiling is reported. With consecutive references (`--dump-xyz 1`) it
also measures the codec-added frame-to-frame change (temporal PSNR). The worst frames are
reviewed by eye in a click-to-flicker A/B viewer on the 384×384 region with the largest
error (2×, no smoothing), with amplified difference maps and whole-frame previews.

| | 4K, 90 frames across the clip | 2K, 90 frames | 4K, hardest 1 s (24 consecutive frames) |
|---|---|---|---|
| VMAF loss (mean / worst) | 0.06 / 1.09 | 0.08 / 0.45 | 0.95 / 2.04 |
| SSIM Y' (mean / worst) | 0.9952 / 0.9809 | 0.9992 / 0.9943 | 0.9818 / 0.9761 |
| XPSNR Y' mean | 59.1 dB | 67.6 dB | 49.0 dB |
| PSNR X'Y'Z' (mean / worst) | 55.9 / 41.6 dB | 65.3 / 47.0 dB | 42.0 / 38.4 dB |
| Temporal PSNR | – | – | 40.5 dB (≈ PSNR − 3 dB: uncorrelated intra-frame noise, as expected) |

The worst frames sit at the end of the clip (frames 632–664 at 4K), the most detailed
content under the fixed 230 Mbit/s budget. The metrics are guides; the flicker viewer is
the check that matters, ideally followed by viewing on a DCI-calibrated display.

## Comparison with OpenJPEG's DCI encoder

Identical X'Y'Z' frames (nine spread across the clip at each resolution), OpenJPEG 2.5.4
`opj_compress -cinema4K` / `-cinema2K 24 -threads ALL_CPUS` vs GAIL at the same
250 Mbit/s cap, both decoded with OpenJPEG:

* Mean PSNR: 4K OpenJPEG 60.38 dB, GAIL 60.43 dB (+0.05); 2K 70.18 vs 70.20 dB (+0.03).
  GAIL is equal or better on 17 of 18 frames at slightly smaller sizes.
* OpenJPEG exceeds the 1,302,083-byte limit on several frames (up to 1,302,099 bytes);
  GAIL never does.
* OpenJPEG: 1.5 fps (4K) / 5.3 fps (2K) on 24 CPU threads; GAIL: 183 / 407 fps on two GPUs.

## Playback on cinema hardware

4K DCPs containing GAIL-encoded picture (SMPTE, 230 Mbit/s, with audio) have been played
back on the following equipment:

| Kind | Tested |
|---|---|
| Projectors | Christie Solaria One, Christie CP2210, Christie CP2220, NEC NC900C-A |
| Theatre management system (LMS) | ScreenWriter Plus 2.7.5-1983 |
| Integrated media block (IMB) | GDC SR-1000 |

This is a list of equipment that has been tried, not a certification: it does not cover
every server, firmware version or content type. Reports from other systems, working or not,
are welcome.

## Not verified here

Other servers and projectors than those above. Compatibility evidence otherwise consists of
the standards checks above, two independent decoders, libdcp's `dcpverify` and
DCP-o-matic's verification. The DCP-o-matic check runs its verification call through a
headless harness rather than clicking through the GUI.
