# Using GAIL as a library

`libgail` is GAIL's encoder as a C library: one header, [include/gail/gail.h](../include/gail/gail.h),
and `libgail.so` (Linux), `libgail.dll` (Windows), `libgail.dylib` (macOS), or the static
`libgail.a`. Every packet it
returns is a complete, DCI-compliant JPEG 2000 codestream (SMPTE ST 428-1 / ST 429-4), ready
to wrap into a DCP.

Ready-made integrations:

* **FFmpeg:** a `libgail` encoder; see [integrations/ffmpeg](../integrations/ffmpeg/README.md).

## Supported hardware

An encoder created with `GAIL_BACKEND_AUTO` uses CUDA when it finds a usable NVIDIA GPU,
otherwise Vulkan when it finds a usable GPU of any make, otherwise the CPU. A GPU that
fails a requirement below is skipped and reported at `GAIL_LOG_WARNING`; only an explicit
backend with no usable device fails (`GAIL_ERROR_UNSUPPORTED`).

| Backend | Needs | Tested on |
|---|---|---|
| **CUDA** | An NVIDIA GPU of the **Turing generation or newer**: GeForce GTX 16 / RTX 20, 30, 40, 50, the Quadro / RTX A / RTX PRO and data-centre parts from the T4 on. An NVIDIA **driver 580 or newer** (what CUDA 13 requires). Nothing else: the CUDA runtime is inside the library. | RTX 3090 (Ampere), RTX 5060 Ti (Blackwell), Linux; MSVC build on Windows |
| **Vulkan** | A GPU with a **Vulkan 1.3** driver and a compute queue, 32-wide subgroups (or subgroup size control that allows 32, as AMD's wave64 parts offer), and the core features `shaderInt64`, `shaderInt16`, `shaderInt8`, 8- and 16-bit storage buffers, `bufferDeviceAddress`, `timelineSemaphore` and `synchronization2`. NVIDIA: Pascal (GTX 10) or newer. AMD RDNA and Intel Arc / Xe drivers advertise all of this. `VK_EXT_memory_budget` and `VK_EXT_pci_bus_info` are used when present (frames-in-flight sizing, device IDs). | RTX 3090 and RTX 5060 Ti, Linux and Windows (under Wine). **AMD and Intel are untested**: the shaders use only core Vulkan 1.3, so they are expected to work; reports welcome |
| **CPU** | Any x86-64 or arm64 CPU. Multithreaded reference implementation, bit-exact with the GPU backends. | Linux, Windows, macOS |

* **Older NVIDIA GPUs or drivers.** Maxwell, Pascal and Volta (GTX 900 / 10, TITAN V) and
  drivers before 580 cannot run CUDA 13, so the library uses them through Vulkan, at
  roughly half the CUDA throughput at 4K. Driver 580 is the last series for Pascal.
* **CUDA architectures.** The library carries native code for sm_75 (Turing), sm_86
  (Ampere GeForce), sm_89 (Ada) and sm_120 (Blackwell GeForce), plus PTX for each, so other
  supported parts (A100, Hopper, data-centre Blackwell) compile the PTX on first use, which
  takes a few seconds once per driver. `GAIL_CUDA_ARCHS` changes the list at build time.
* **Device memory.** One frame in flight takes about 0.7 GB at 4K and 0.2 GB at 2K; the
  encoder starts as many as fit (3 by default) and runs with one if that is all there is,
  so a 2 GB card works at 4K, slowly. See "Device memory and frames in flight" below.
* **macOS** is CPU-only until the Vulkan backend has been tried on MoltenVK (see
  "Building: macOS").

## Building

### Linux

```bash
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build build -j
ctest --test-dir build
sudo cmake --install build
```

This installs:

* `include/gail/gail.h`;
* `lib/libgail.so.<version>`, `libgail.a`;
* `lib/pkgconfig/gail.pc`;
* `lib/cmake/gail/` (`find_package(gail)`, target `gail::gail`);
* `gail-enc` and `gail-check`.

Build requirements:

* a C++20 compiler and CMake ≥ 3.20;
* the Vulkan headers and `glslc` for the Vulkan backend;
* the CUDA toolkit for the optional CUDA backend.

At run time, `libgail.so` needs only the C/C++ runtime:

* It opens the Vulkan loader itself when one is present.
* It links the CUDA runtime statically, so it needs only the NVIDIA driver. CUDA 13
  needs driver 580 or newer and a Turing or newer GPU (GTX 16 / RTX 20 and later; the
  kernels are built for sm_75, 86, 89 and 120 with PTX for the rest). Older NVIDIA GPUs,
  and older drivers, are used through Vulkan instead.
* Without a usable GPU it falls back to the CPU.

| CMake option | Default | |
|---|---|---|
| `GAIL_BUILD_SHARED` / `GAIL_BUILD_STATIC` | ON / ON | which libraries to build |
| `GAIL_ENABLE_VULKAN` | ON | Vulkan backend (NVIDIA, AMD, Intel) |
| `GAIL_ENABLE_CUDA` | AUTO | CUDA backend (NVIDIA), if `nvcc` is found |
| `GAIL_BUILD_TOOLS`, `GAIL_BUILD_TESTS` | ON | command-line tools and tests |
| `GAIL_NATIVE_ARCH` | OFF | `-march=native`, for builds that only run on this machine |

### Windows

**MinGW-w64.** FFmpeg's Windows builds use this toolchain, usually cross-compiled from
Linux:

```bash
cmake -S . -B build-win -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-x86_64.cmake \
      -DGAIL_VULKAN_INCLUDE_DIR=/path/with/vulkan-headers
cmake --build build-win
```

* `GAIL_VULKAN_INCLUDE_DIR` is any directory containing `vulkan/vulkan.h`.
* The result is `libgail.dll` with import library `libgail.dll.a`.
* Every binary is linked statically against the MinGW runtime, so `libgail.dll` imports
  only `KERNEL32.dll` and the Windows C runtime. There are no other DLLs to ship.
* Tested under Wine, where every test passes on the GPUs through Vulkan.
* CUDA cannot be built with MinGW (NVIDIA's compiler needs MSVC on Windows), so these
  builds use Vulkan, which works on NVIDIA, AMD and Intel GPUs.

**MSVC.** This is the only way to get the CUDA backend on Windows. It is verified with Visual
Studio 2022 Build Tools (MSVC 19.44), CMake 3.31 and Ninja as bundled with them, the Vulkan
SDK 1.4, and CUDA 13.4:

```bat
"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set VULKAN_SDK=C:\VulkanSDK
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGAIL_ENABLE_CUDA=ON ^
      -DCMAKE_CUDA_COMPILER=%CUDA_PATH%/bin/nvcc.exe -DCUDAToolkit_ROOT=%CUDA_PATH%
cmake --build build
ctest --test-dir build
```

* **CUDA:** either the full toolkit installer or NVIDIA's redistributable zips work. With
  the zips, unpack `cuda_nvcc`, `cuda_cudart`, `cuda_crt`, `libnvvm` and `cccl` into one
  folder. CMake's CUDA package needs `cccl`'s headers, even though GAIL does not use them.
* **Output:** `gail.dll` (import library `gail.lib`) and the static `gail_static.lib`.
* **Exports:** the DLL exports only the `gail_*` functions.
* **Dependencies:** Windows itself and the standard Visual C++ runtime (`MSVCP140` /
  `VCRUNTIME140`, from the Visual C++ Redistributable).
* **Tests:** both test suites pass, on a machine without a GPU (CPU fallback).

### macOS

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGAIL_ENABLE_VULKAN=OFF
ninja -C build && ctest --test-dir build
```

* **Toolchain:** Apple clang from the Xcode Command Line Tools, plus CMake and Ninja
  (Homebrew, or `pip3 install --user cmake ninja`). There are no other dependencies.
* **Output:** `libgail.dylib`, exporting only the `gail_*` functions, depending only on
  `libc++` and `libSystem`; and the static `libgail.a`.
* **Tested:** the CPU backend, on an Intel macOS 26 virtual machine: both test suites
  pass, with no compiler warnings. `low_priority` uses the Utility QoS class.
* **GPU:** CUDA does not exist on macOS. The Vulkan backend compiles against the Vulkan
  SDK (`-DGAIL_ENABLE_VULKAN=ON` with `glslc` on the path) and loads `libvulkan.1.dylib`
  or `libMoltenVK.dylib` at run time, but it has not been run on a Mac: MoltenVK must
  expose the subgroup and 64-bit atomic features the shaders use. Until someone with an
  Apple-silicon Mac tries it, treat macOS as CPU-only. Reports welcome.

## Continuous integration

[.github/workflows/ci.yml](../.github/workflows/ci.yml) builds and tests every push and pull
request:

| Job | Runner | Builds | Tests |
|---|---|---|---|
| `linux` | Ubuntu 24.04, GCC | CUDA + Vulkan + CPU | both suites; install, then a program built against the installed library through pkg-config |
| `windows-msvc` | Windows Server 2022, MSVC | CUDA + Vulkan + CPU | both suites; install |
| `windows-mingw` | Ubuntu 24.04, MinGW-w64 cross | Vulkan + CPU | both suites under Wine |
| `macos` | macOS 15, Apple clang (arm64) | CPU | both suites; install, then a pkg-config consumer |
| `ffmpeg` | Ubuntu 24.04 | FFmpeg with the `libgail` encoder | encodes a test pattern; `gail-check` verifies it |

* **No GPU on the runners:** the CUDA and Vulkan backends are compiled in every job, but
  tests run on the CPU backend, and the GPU tests skip themselves. GPU coverage comes from
  running `ctest` on a machine with GPUs.
* **CUDA:** `ci/fetch_cuda.py` downloads just the CUDA parts needed to compile (NVIDIA's
  redistributable archives, checked against the SHA-256 in NVIDIA's manifest). This needs
  no installer and no driver, and also works for local builds.
* **Pinned versions:** CUDA 13.4.92, Vulkan SDK 1.4.363.0 and FFmpeg n9.0.2, set at the top
  of the workflow. Downloads are cached between runs.
* **Artifacts:** each build job uploads its installed library (Linux, Windows MSVC,
  Windows MinGW, macOS).

## Linking

```bash
cc app.c $(pkg-config --cflags --libs gail)          # shared
cc app.c $(pkg-config --cflags --libs --static gail)  # static (same header on every platform)
```

### Versions

The `GAIL_VERSION_*` macros in `gail.h` hold the release version. Every build also works out
its full version from git, which `gail_version_string()` reports and every frame's COM marker
carries:

| Build | Full version |
| --- | --- |
| Clean checkout of the tag `v1.1.0` | `1.1.0` |
| 3 commits after that tag | `1.1.0+3.gabc1234` |
| The same, with uncommitted changes | `1.1.0+3.gabc1234.dirty` |
| No git, such as a source archive | `1.1.0` |

To release, run this on a clean tree. It updates `gail.h`, commits, and tags:

```bash
scripts/release.py patch           # or minor, major, or an exact version such as 1.2.0
git push origin HEAD --follow-tags  # CI fails if the tag and gail.h disagree
```

### Compatibility

The shared library exports only the `gail_*` functions, with symbol version `GAIL_1`
on Linux. `gail_version()` reports the version actually loaded. Within version 1:

* `gail_config` only grows at its end. `gail_config_defaults()` records the header's size in
  `struct_size`, so a newer library accepts a config from an older header. A config from
  a newer header is accepted as long as it leaves the newer settings at their defaults.
* Every other struct keeps its size, and new fields take the place of `reserved` words.
  Zero them: start from `gail_config_defaults()` / `gail_color_preset()` and write
  `gail_frame f = {0}`. The encoder rejects non-zero reserved words.
* Bindings for other languages call `gail_config_init(&cfg, sizeof cfg)`, the exported
  function behind `gail_config_defaults()`.

`gail_encoder_cancel()` stops an encode at once (a Cancel button): blocked calls return
`GAIL_ERROR_CANCELLED` and destroying the encoder then only waits for the frames on the
devices.

## Calling patterns

| | How | Typical caller |
|---|---|---|
| Pull | `gail_encoder_send()` frames; `gail_encoder_receive()` packets in order; `send(NULL)` ends; receive until `GAIL_EOF` | FFmpeg-style encoders |
| Callback | `config.on_packet`, called in order from an encoder thread; `gail_encoder_finish()` | pipelines that write files |
| Blocking | `gail_encoder_encode()` from any number of threads; each gets its own frame's codestream | apps with their own pool of worker threads |

* **Ownership:** packets from `receive` / `encode` belong to the caller. Release them with
  `gail_packet_free()`, which uses the library's own heap (this matters for a Windows DLL).
* **Return values:** every call returns a `gail_status`, and `gail_last_error()` explains
  failures. Nothing is thrown across the C boundary.
* **Logging:** `config.log` receives every message the library has (devices, colour,
  non-DCI frame rates, non-compliant frames, a GPU that could not be set up, a frame
  redone on the CPU). The library never writes to stdout or stderr: with `log` NULL
  it is silent. `GAIL_LOG_WARNING` is worth showing to users; `GAIL_LOG_INFO` describes
  each device and the colour settings, as `gail-enc -v` prints them.

## Choosing GPUs

By default an encoder uses every GPU of its backend (CUDA if there are NVIDIA GPUs, else
Vulkan, else the CPU). To let users choose:

```c
gail_device_info devs[16];
int n = gail_enumerate_devices(GAIL_BACKEND_AUTO, devs, 16);  /* cheap: nothing is set up */
for (int i = 0; i < n && i < 16; ++i)
    printf("%s  %s  %s  %.0f GB\n", devs[i].name, devs[i].uuid, devs[i].pci_bus_id,
           devs[i].memory_bytes / 1073741824.0);
/* NVIDIA GeForce RTX 3090  GPU-d14c72ba-5566-4a1e-adca-5212d5f9123d  0000:0c:00.0  24 GB */

const char* chosen[] = {"GPU-d14c72ba-5566-4a1e-adca-5212d5f9123d"};
cfg.device_ids = chosen;      /* store these, not indices */
cfg.num_device_ids = 1;
```

* **UUIDs** are the ones `nvidia-smi -L` prints, and the same GPU has the same UUID under
  CUDA and Vulkan.
* **Matching:** an ID matches a device by UUID (the `GPU-` prefix and case are optional) or
  by PCI bus ID (`0000:0c:00.0`, or `0c:00.0`).
* **Why IDs, not indices:** IDs stay valid when GPUs are added or drivers change. Indices
  (`cfg.devices`) can change.
* **Unknown IDs:** creating an encoder with one fails with `GAIL_ERROR_UNSUPPORTED`, and
  `gail_last_error()` lists the available devices.
* **As text:** `gail_list_devices()` returns the same list as text, and
  `gail-enc --list-devices` prints it.

## Device memory and frames in flight

A frame in flight holds its own buffers on the GPU: about 0.7 GB at 4K and 0.2 GB at 2K
(CUDA a little more than Vulkan). Users often run the encoder on their only GPU, next to a
browser, a player or another encode, so the encoder does not assume the memory is free:

* **Sizing at start.** The encoder asks the driver how much device memory this process may
  still take (`cudaMemGetInfo` / `VK_EXT_memory_budget`) and starts only as many frames in
  flight as fit after keeping 512 MiB or 6 % of the card free for everyone else. Fewer
  frames than `frames_in_flight` (default 3) means lower throughput, never a failure.
* **Hard refusals are soft.** A frame whose memory the driver refuses anyway is dropped; the
  frames already set up carry on. If not even one fits, `GAIL_BACKEND_AUTO` falls back to
  the CPU backend (slow but complete); an explicit backend fails with `GAIL_ERROR_NOMEM`.
* **Pressure during the run.** Twice a second one frame per device checks the free memory.
  When another process has pushed it under half the headroom, that frame gives up its
  buffers so the other process is not the one to run out. One frame per device always
  stays; frames are not re-added within a run.
* **Caps.** `max_device_memory` (bytes per device) caps what the frames in flight may take,
  for hosts that know how much they need for themselves. Every reduction is reported at
  `GAIL_LOG_WARNING` with the numbers.

The same logic drives `gail-enc` / `gail-bench` (`--slots`, `--device-memory MB`).

## CPU threads and priority

* `cpu_threads` sets the CPU backend's worker threads (also used when an auto-mode encoder
  falls back to the CPU). 0 means every core the process may run on, read from the affinity
  mask (`taskset`, cgroups, Windows job objects), not the machine's core count. The CPU
  backend keeps `frames_in_flight` frames going on that one pool (default 2), so a host
  with its own thread pool should give the encoder only the cores it wants to spend.
* `low_priority` runs the encoder's threads below normal priority (nice +10 on Linux, below
  normal on Windows, Utility QoS on macOS): the CPU backend's workers, and the per-frame threads that do rate
  control and Tier-2 behind the GPU backends (one per frame in flight per device, each a
  few ms of CPU per frame). Throughput on an idle machine is unchanged; next to a busy
  desktop the encoder yields first.
* The GPU backends' remaining CPU use is the host's: in `gail-enc` it is FFmpeg decoding.

`gail-enc` / `gail-bench`: `--threads N`, `--low-priority`, and `--slots N` (applies to the
CPU backend too; default 2 there).

## Input

| `gail_pixel_format` | Layout | Colour |
|---|---|---|
| `GAIL_PIX_XYZ12` | interleaved 16-bit, 12-bit X'Y'Z' codes in the low bits (libdcp) | encoded exactly |
| `GAIL_PIX_XYZ12_MSB` | the same, codes in the high bits (FFmpeg `xyz12le`) | encoded exactly |
| `GAIL_PIX_RGB48` | interleaved 16-bit R'G'B' | `config.color` |
| `GAIL_PIX_YUV420P`, `…P10`, `…P12` | planar Y'CbCr 4:2:0 (H.264, HEVC) | `config.color` + `input.matrix`, range, chroma location |
| `GAIL_PIX_YUV422P`, `…P10`, `…P12` | planar Y'CbCr 4:2:2 (ProRes 422, DNxHR HQX) | `config.color` + `input.matrix`, range, chroma location |
| `GAIL_PIX_YUV444P`, `…P10`, `…P12` | planar Y'CbCr 4:4:4 (ProRes 4444, DNxHR 444) | `config.color` + `input.matrix`, range |
| `GAIL_PIX_GBRP`, `…P10`, `…P12`, `…P16` | planar full-range R'G'B', planes G', B', R' (DPX, TIFF) | `config.color` |

Planar samples deeper than 8 bits are 16-bit words with the value in the low bits, as
FFmpeg's `…le` formats.

* **Container:** the container is set by `profile` + `container` (flat, scope, full). A
  picture smaller than the container (`input.width` / `height`) is centred with black
  borders; scaling is the caller's job.
* **Planar input:** Y'CbCr and planar R'G'B' are converted on the GPU, so an app can hand
  over its decoder's own layout instead of converting it on the CPU first.
* **Colour:** by default the handling matches DCP-o-matic. See [COLOR.md](COLOR.md) for
  the options.
* **Frame rates:** DCI frame rates are 24, 25, 30, 48, 50 and 60. At other rates the
  library warns, because the bitrate is shared over more frames.
* **3D:** `config.stereo` gives each eye half the frame budget, as DCI requires.

## Example

[tests/test_api.c](../tests/test_api.c) exercises every mode and format and is the
reference example. The shortest complete use:

```c
#include <gail/gail.h>

gail_config cfg;
gail_config_defaults(&cfg);
cfg.profile = GAIL_PROFILE_4K;
cfg.input.format = GAIL_PIX_XYZ12;
gail_encoder* enc;
if (gail_encoder_create(&cfg, &enc) != GAIL_OK) { puts(gail_last_error()); return 1; }
for (each frame) {
    gail_frame f = {{xyz}, {0}, NULL};
    gail_packet pkt;
    if (gail_encoder_encode(enc, &f, &pkt) != GAIL_OK) break;
    fwrite(pkt.data, 1, pkt.size, out);
    gail_packet_free(&pkt);
}
gail_encoder_destroy(enc);
```
