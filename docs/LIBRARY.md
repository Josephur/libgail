# Using GAIL as a library

`libgail` is GAIL's encoder and decoder as a C library: one header, [include/gail/gail.h](../include/gail/gail.h),
and `libgail.so` (Linux), `libgail.dll` (Windows), `libgail.dylib` (macOS), or the static
`libgail.a`. Every packet the encoder
returns is a complete, DCI-compliant JPEG 2000 codestream (SMPTE ST 428-1 / ST 429-4), ready
to wrap into a DCP; the decoder turns such codestreams back into X'Y'Z' pictures
([Decoding](#decoding)).

Ready-made integrations:

* **FFmpeg:** a `libgail` encoder and decoder; see [integrations/ffmpeg](../integrations/ffmpeg/README.md).

## Supported hardware

An encoder created with `GAIL_BACKEND_AUTO` uses CUDA when it finds a usable NVIDIA GPU,
otherwise Vulkan when it finds a usable GPU of any make, otherwise the CPU. A GPU that
fails a requirement below is skipped and reported in the log: at `GAIL_LOG_INFO` when it
is a GPU the CUDA plugins cannot run, at `GAIL_LOG_WARNING` otherwise. Only an explicit
backend with no usable device fails (`GAIL_ERROR_UNSUPPORTED`).

| Backend | Needs | Tested on |
|---|---|---|
| **CUDA** | An NVIDIA GPU of the **Maxwell generation or newer** (compute capability 5.0 and up): GeForce GTX 750 Ti / 900 / 10 / 16 and RTX 20, 30, 40, 50, the Quadro / RTX A / RTX PRO and data-center parts of those generations. The [CUDA plugin](#cuda-plugins) for the GPU's generation next to the library, and the **NVIDIA driver** that plugin needs. Nothing else: each plugin has its CUDA runtime inside. | RTX 3090 (Ampere), RTX 5060 Ti (Blackwell), Linux; MSVC build on Windows |
| **Vulkan** | A GPU with a **Vulkan 1.3** driver and a compute queue, 32-wide subgroups (or subgroup size control that allows 32, as AMD's wave64 parts offer), and the core features `shaderInt64`, `shaderInt16`, `shaderInt8`, 8- and 16-bit storage buffers, `bufferDeviceAddress`, `timelineSemaphore` and `synchronization2`. NVIDIA: Pascal (GTX 10) or newer. AMD RDNA and Intel Arc / Xe drivers advertise all of this. `VK_EXT_memory_budget` and `VK_EXT_pci_bus_info` are used when present (frames-in-flight sizing, device IDs). | RTX 3090 and RTX 5060 Ti, Linux and Windows (under Wine). **AMD and Intel are untested**: the shaders use only core Vulkan 1.3, so they are expected to work; reports welcome |
| **CPU** | Any x86-64 or arm64 CPU. Multithreaded reference implementation, bit-exact with the GPU backends. | Linux, Windows, macOS |

* **Older NVIDIA drivers, missing plugins.** A GPU whose plugin is not next to the
  library, or whose driver is too old for that plugin, is not a CUDA device. It is used
  through Vulkan, which is slower (about half the CUDA throughput at 4K on the GPUs of
  [BENCHMARKS.md](BENCHMARKS.md)). See [CUDA plugins](#cuda-plugins).
* **CUDA architectures.** `gail_cuda13` carries native code for sm_75 (Turing), sm_86
  (Ampere GeForce), sm_89 (Ada) and sm_120 (Blackwell GeForce), plus PTX for each, so other
  supported parts (A100, Hopper, data-center Blackwell) compile the PTX on first use, which
  takes a few seconds once per driver. `gail_cuda12` carries sm_50 and sm_52 (Maxwell),
  sm_60 and sm_61 (Pascal) and sm_70 (Volta). `GAIL_CUDA13_ARCHS` and `GAIL_CUDA12_ARCHS`
  change the lists at build time.
* **Device memory.** One frame in flight takes about 0.7 GB at 4K and 0.2 GB at 2K; the
  encoder starts as many as fit (3 by default) and runs with one if that is all there is,
  so a 2 GB card works at 4K, slowly. See "Device memory and frames in flight" below.
* **macOS** is CPU-only until the Vulkan backend has been tried on MoltenVK (see
  "Building: macOS").

### CUDA plugins

The shared library links no CUDA. The CUDA backend is built once per CUDA toolkit, each
time into a plugin, a shared library of its own with that toolkit's runtime inside, which
libgail loads at run time. There are two because CUDA 13 no longer builds for the GPU
generations before Turing: Maxwell, Pascal and Volta GPUs use the CUDA 12.9 plugin, Turing
and newer the CUDA 13.4 plugin.

| Plugin file, Windows / Linux | CUDA toolkit | GPU generations | Compute capability | Example cards | Minimum NVIDIA driver, Windows | Minimum NVIDIA driver, Linux |
|---|---|---|---|---|---|---|
| `gail_cuda12.dll` / `libgail_cuda12.so` | 12.9 | Maxwell, Pascal, Volta | 5.0 – 7.0 | GTX 750 Ti, GTX 950 – 980 Ti, GT 1030, GTX 1050 – 1080 Ti, Titan X / Xp / V, Quadro M and P series, Tesla M40, P4, P40, P100, V100 | 576.57 | 575.57.08 |
| `gail_cuda13.dll` / `libgail_cuda13.so` | 13.4 | Turing, Ampere, Ada, Blackwell | 7.5 and up | GTX 1650 – 1660 Ti, RTX 20, 30, 40 and 50 series, Quadro RTX, RTX A and RTX PRO series, T4 | 580 | 580 |

* **One plugin per GPU.** Which plugin a GPU gets is fixed by its generation: compute
  capability 7.5 and up is `gail_cuda13`, anything older `gail_cuda12`. Neither stands in
  for the other. The GTX 16 series is Turing, so it uses `gail_cuda13`. With GPUs of both
  kinds in one machine, both plugins run side by side in one encoder.
* **Drivers.** The table gives the versions from NVIDIA's CUDA Toolkit release notes: the
  driver that goes with CUDA 12.9 (Update 1), and the oldest driver CUDA 13.x applications
  run on. NVIDIA has named R580 as the last driver branch for Maxwell, Pascal and Volta;
  those GPUs stay on a 580-series driver, which runs `gail_cuda12`. With a driver too old
  for a plugin, the plugin's CUDA runtime does not start and CUDA does not offer its GPUs.
* **Falling back.** A GPU that no plugin can run, a plugin file that is missing, and a
  machine with no NVIDIA driver are not errors. `GAIL_BACKEND_AUTO` uses CUDA for the GPUs
  it can run. When it can run none of them, or not every GPU named in `device_ids`, the
  GPUs are used through Vulkan (see the Vulkan row above for what that needs), and
  without a usable GPU the CPU does the work. An encoder uses one backend, so next to GPUs
  that CUDA runs, a GPU it cannot run is left out. What was left out, and why, is logged
  at `GAIL_LOG_INFO`. Only `GAIL_BACKEND_CUDA` asked for by name fails, with
  `GAIL_ERROR_UNSUPPORTED` and the reason in `gail_last_error()`.
  Which backend runs changes the speed, not the encoder: the CUDA plugins, Vulkan and the
  CPU all write DCI-compliant codestreams, and their Tier-1 coding is bit-identical (see
  [VALIDATION.md](VALIDATION.md)).
* **Shipping.** Applications ship both plugin files in the same directory as the gail
  library: `gail_cuda13.dll` and `gail_cuda12.dll` next to `gail.dll`,
  `libgail_cuda13.so` and `libgail_cuda12.so` next to `libgail.so`. That directory is the
  only place the library looks; the `PATH`, the working directory and the system's library
  search order play no part. The release archives hold them there already (`bin/` on
  Windows, `lib/` on Linux). The MinGW-w64 and macOS builds have no CUDA backend and no
  plugins.
* **Checking.** `gail-enc --list-devices` shows which plugin serves each GPU, which
  plugins are loaded, and any GPU CUDA cannot use, with the reason.
  `gail_version_string()` names the plugins loaded, as in
  `1.1.0 (cuda vulkan cpu; gail_cuda13 gail_cuda12)`.
* **Listing devices.** A GPU is offered as a CUDA device when its plugin holds machine
  code for its compute capability (the same major version, the same or a lower minor), or
  PTX for an earlier one. Listing goes by that alone and sets nothing up on the GPU:
  `gail_enumerate_devices()` starts each plugin's CUDA runtime, which takes a few tens of
  milliseconds the first time in a process, and leaves no device memory in use. Whether
  the kernels really load on the GPU is found out when an encoder or decoder sets it up. A
  GPU where they do not, such as one whose memory is full, is then left out with the
  reason at `GAIL_LOG_WARNING`: `GAIL_BACKEND_AUTO` carries on with the other CUDA GPUs,
  or with Vulkan when none is left, and `GAIL_BACKEND_CUDA` fails.
* **Static library.** `libgail.a` cannot carry plugins, so it has one backend linked in:
  the CUDA 13 one when the build had both toolkits. It serves the GPUs that backend has
  code for.

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
* `lib/libgail_cuda13.so`, `lib/libgail_cuda12.so`, the [CUDA plugins](#cuda-plugins), for
  each CUDA toolkit the build had;
* `lib/pkgconfig/gail.pc`;
* `lib/cmake/gail/` (`find_package(gail)`, target `gail::gail`);
* `gail-enc` and `gail-check`.

Build requirements:

* a C++20 compiler and CMake ≥ 3.20;
* the Vulkan headers and `glslc` for the Vulkan backend;
* for the optional CUDA backend, the CUDA 13 toolkit (the `gail_cuda13` plugin), the CUDA
  12.9 toolkit (`gail_cuda12`), or both.

At run time, `libgail.so` needs only the C/C++ runtime:

* It opens the Vulkan loader itself when one is present.
* It loads the [CUDA plugins](#cuda-plugins) that are in its own directory. Each has its
  CUDA runtime linked in statically, so CUDA needs only the NVIDIA driver. Without a
  plugin for a GPU, or with too old a driver, Vulkan is used instead.
* Without a usable GPU it falls back to the CPU.

| CMake option | Default | |
|---|---|---|
| `GAIL_BUILD_SHARED` / `GAIL_BUILD_STATIC` | ON / ON | which libraries to build |
| `GAIL_ENABLE_VULKAN` | ON | Vulkan backend (NVIDIA, AMD, Intel) |
| `GAIL_ENABLE_CUDA` | AUTO | CUDA backend (NVIDIA), if a CUDA toolkit is found |
| `GAIL_CUDA13_ROOT`, `GAIL_CUDA12_ROOT` | | the CUDA 13 and CUDA 12.9 toolkits to build `gail_cuda13` and `gail_cuda12` with |
| `GAIL_CUDA13_ARCHS`, `GAIL_CUDA12_ARCHS` | `75;86;89;120`, `50;52;60;61;70` | CUDA architectures of each plugin |
| `GAIL_CUDA12_HOST_COMPILER` | | a C++ compiler for `gail_cuda12`, when CUDA 12.9 does not accept the project's |
| `GAIL_BUILD_TOOLS`, `GAIL_BUILD_TESTS` | ON | command-line tools and tests |
| `GAIL_NATIVE_ARCH` | OFF | `-march=native`, for builds that only run on this machine |

**Both CUDA plugins.** With neither `GAIL_CUDA13_ROOT` nor `GAIL_CUDA12_ROOT` set, the
toolkit CMake finds (`nvcc` on the `PATH`, `CUDA_PATH`, `CMAKE_CUDA_COMPILER`) is used, and
builds the one plugin it is for. To build both, name the two toolkits. `ci/fetch_cuda.py`
fetches what is needed of either:

```bash
python3 ci/fetch_cuda.py cuda13 --manifest 13.4.3
python3 ci/fetch_cuda.py cuda12 --manifest 12.9.2
cmake -S . -B build -DGAIL_CUDA13_ROOT=$PWD/cuda13 -DGAIL_CUDA12_ROOT=$PWD/cuda12
```

* The plugins are built from the same sources, `src/cuda/*.cu`, and end up next to
  `libgail.so` in the build tree and in `lib/` when installed.
* CUDA 12.9 accepts GCC up to version 14. On a system whose compiler is newer, give the
  CUDA 12.9 plugin an older one with `-DGAIL_CUDA12_HOST_COMPILER=g++-14`; the rest of the
  build keeps the system's compiler.
* The static library links the CUDA 13 backend when both toolkits are there.

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
      -DGAIL_CUDA13_ROOT=C:/cuda13 -DGAIL_CUDA12_ROOT=C:/cuda12
cmake --build build
ctest --test-dir build
```

* **CUDA:** `GAIL_CUDA13_ROOT` and `GAIL_CUDA12_ROOT` are a CUDA 13 and a CUDA 12.9
  toolkit, each built into its [plugin](#cuda-plugins); with only one of them, only that
  plugin is built. Either the full toolkit installer or NVIDIA's redistributable zips work
  (`ci/fetch_cuda.py` fetches and unpacks the zips). By hand, unpack `cuda_nvcc`,
  `cuda_cudart`, `cuda_crt`, `libnvvm` and `cccl` of CUDA 13 into one folder, and
  `cuda_nvcc`, `cuda_cudart` and `cuda_cccl` of CUDA 12.9 into another. CMake's CUDA
  package needs the `cccl` headers, even though GAIL does not use them.
* **Output:** `gail.dll` (import library `gail.lib`), the plugins `gail_cuda13.dll` and
  `gail_cuda12.dll` next to it, and the static `gail_static.lib`.
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
| `linux` | Ubuntu 24.04, GCC | CUDA (both plugins) + Vulkan + CPU | both suites; both plugins load; install, then a program built against the installed library through pkg-config, which finds the plugins |
| `sanitize` | Ubuntu 24.04, GCC with AddressSanitizer and UndefinedBehaviorSanitizer | CPU | the decoder's tests, damaged codestreams among them, and the API test |
| `windows-msvc` | Windows Server 2022, MSVC | CUDA (both plugins) + Vulkan + CPU | both suites; both plugins load; install |
| `windows-mingw` | Ubuntu 24.04, MinGW-w64 cross | Vulkan + CPU | both suites under Wine |
| `macos` | macOS 15, Apple clang (arm64) | CPU | both suites; install, then a pkg-config consumer |
| `ffmpeg` | Ubuntu 24.04 | FFmpeg with the `libgail` encoder and decoder | encodes a test pattern, which `gail-check` verifies; decodes it again and compares with FFmpeg's own decoder |

* **No GPU on the runners:** the CUDA and Vulkan backends are compiled in every job, but
  tests run on the CPU backend, and the GPU tests skip themselves. GPU coverage comes from
  running `ctest` on a machine with GPUs.
* **CUDA:** `ci/fetch_cuda.py` downloads just the CUDA parts needed to compile (NVIDIA's
  redistributable archives, checked against the SHA-256 in NVIDIA's manifest), once for
  CUDA 13 and once for CUDA 12.9. This needs no installer and no driver, and also works
  for local builds. The plugins' loader is tested with stand-in plugins that need no GPU
  (`gail-tests plugins`).
* **Pinned versions:** CUDA 13.4.92 and 12.9.86, Vulkan SDK 1.4.363.0 and FFmpeg n9.0.2, set
  at the top of the workflow. Downloads are cached between runs.
* **Artifacts:** each build job uploads its installed library (Linux, Windows MSVC,
  Windows MinGW, macOS).

## Linking

```bash
cc app.c $(pkg-config --cflags --libs gail)          # shared
cc app.c $(pkg-config --cflags --libs --static gail)  # static (same header on every platform)
```

An application linked against the shared library ships the [CUDA plugins](#cuda-plugins)
next to it. The static library has one CUDA backend linked in and needs no plugins.

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

The COM marker's text is `GAIL`, the full version and, after a space, the kind of device that
encoded that frame:

| Encoded by | COM text |
| --- | --- |
| CPU backend | `GAIL 1.1.0` |
| Vulkan | `GAIL 1.1.0 vulk` |
| CUDA | `GAIL 1.1.0 cuda13.4`: the CUDA toolkit (major.minor) of the plugin that served the GPU, or of the backend a static library links in |

The tag is per frame, so a run over several devices, or one that falls back to the CPU, labels
each frame with the device that encoded it. The frame budget always allows for the longest tag
(9 characters), so the tag never changes a frame's coded data: frames encoded from the same
coefficients differ only in the COM marker.

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
* `gail_decoder_config` follows the same rules as `gail_config`, with
  `gail_decoder_config_defaults()` / `gail_decoder_config_init()`; `gail_packet` and
  `gail_image` handed to the decoder have their `reserved` words zero.

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
* **Logging:** `config.log` receives every message the library has (devices, color,
  non-DCI frame rates, non-compliant frames, a GPU that could not be set up, a frame
  redone on the CPU). The library never writes to stdout or stderr: with `log` NULL
  it is silent. `GAIL_LOG_WARNING` is worth showing to users; `GAIL_LOG_INFO` describes
  each device and the color settings, as `gail-enc -v` prints them.

## Choosing GPUs

By default an encoder uses every GPU of its backend (CUDA if there are NVIDIA GPUs, else
Vulkan, else the CPU). To let users choose:

```c
gail_device_info devs[16];
int n = gail_enumerate_devices(GAIL_BACKEND_AUTO, devs, 16);  /* cheap: no encoder is set up */
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
  `gail-enc --list-devices` prints it, with the [CUDA plugin](#cuda-plugins) that serves
  each NVIDIA GPU.
* **GPUs CUDA cannot run** are not among the CUDA devices. When CUDA can run none,
  `GAIL_BACKEND_AUTO` lists the GPUs, and uses them, through Vulkan.

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

| `gail_pixel_format` | Layout | Color |
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
  picture smaller than the container (`input.width` / `height`) is centered with black
  borders; scaling is the caller's job.
* **Planar input:** Y'CbCr and planar R'G'B' are converted on the GPU, so an app can hand
  over its decoder's own layout instead of converting it on the CPU first.
* **Color:** by default the handling matches DCP-o-matic. See [COLOR.md](COLOR.md) for
  the options.
* **Frame rates:** DCI frame rates are 24, 25, 30, 48, 50 and 60. At other rates the
  library warns, because the bitrate is shared over more frames.
* **3D:** `config.stereo` gives each eye half the frame budget, as DCI requires.

## Decoding

A `gail_decoder` takes DCI JPEG 2000 codestreams, one per frame, as a DCP's picture track
file holds them (the caller unwraps the MXF, and decrypts it if it is encrypted), and
returns pictures. It uses the same backends and devices as the encoder, chosen the same
way (`backend`, `devices` / `device_ids`, `frames_in_flight`, `cpu_threads`,
`low_priority`, `log` in `gail_decoder_config`), with the same hardware requirements.
One difference: with no devices named, a decoder uses only the first device, since one GPU
decodes 4K several times faster than it plays; `num_devices = GAIL_ALL_DEVICES` (with
`devices` NULL) uses every device.

| | How | Typical caller |
|---|---|---|
| Pull | `gail_decoder_send()` codestreams; `gail_decoder_receive()` images in order; `send(NULL, NULL)` ends; receive until `GAIL_EOF` | FFmpeg-style decoders, players |
| Callback | `config.on_image`, called in order from a decoder thread; `gail_decoder_finish()` | players with their own display queue |
| Blocking | `gail_decoder_decode()` from any number of threads; each gets its own picture | apps with their own pool of worker threads |

* **What it decodes:** DCI streams: Rsiz 3 or 4, three 12-bit components, one tile, the
  9/7 wavelet, 32×32 code-blocks, one quality layer; GAIL's own and other encoders'.
  `gail_probe()` reads a codestream's main header and reports its size, levels and whether
  it is supported, with the reason in `gail_last_error()` when it is not. Lossless (5/3)
  streams, other bit depths, several tiles or layers and HTJ2K are recognized and refused.
* **Size:** the decoder takes the picture size from the first codestream. A later one of
  another size or with another number of wavelet levels fails with
  `GAIL_ERROR_UNSUPPORTED`; it needs a decoder of its own.
* **Output:** three 16-bit words per pixel, X', Y', Z'. `GAIL_PIX_XYZ12_MSB` (the default)
  has the 12-bit codes in the high bits: FFmpeg's `xyz12le`, what FFmpeg's JPEG 2000
  decoder returns and what players' display code expects, so converting for a monitor and
  scaling work downstream as they do today. `GAIL_PIX_XYZ12` has them in the low bits, the
  layout the encoder takes.
* **Whose buffer:** by default the library allocates the picture and `gail_image_free()`
  releases it (the buffers are reused, so release them and do not keep hundreds). Or the
  caller names its own buffer in `data[0]` with its row length in `stride[0]` (bytes, at
  least 6 × width, any alignment; bytes of a row past the picture are not written) and the
  picture is decoded straight into it: no copy, which is how the FFmpeg decoder fills
  FFmpeg's frames. A buffer handed to `gail_decoder_send()` must stay valid until its image
  has been received, or `gail_decoder_flush()` or `gail_decoder_destroy()` has returned.
* **Reduced size:** `config.reduce = 1` decodes at half size, `2` at a quarter (rounded
  up), by leaving out wavelet levels: much less work, for previews or a 2K display of a 4K
  track.
* **Errors:** codestreams are untrusted input. One that is damaged or unsupported fails on
  its own: `gail_decoder_decode()` returns the status, and in pull or callback mode the
  image arrives in its place in the order with a negative `status` and no picture, with
  the reason in `gail_last_error()`. The decoder carries on. `gail_decoder_error()` is
  only set when a device fails.
* **Seeking:** `gail_decoder_flush()` drops what was sent and not yet received.
  `gail_decoder_cancel()` stops everything at once.
* **Device memory:** a frame in flight takes about 0.26 GB (4K) or 0.07 GB (2K) on a GPU.
  A device that cannot hold the requested number runs fewer, with a warning.
* **Agreement between backends:** the CUDA and Vulkan decoders reproduce the CPU decoder's
  coefficients exactly and its pictures to within one code value (the inverse wavelet is
  float arithmetic and rounds differently on a GPU).

```c
gail_decoder_config cfg;
gail_decoder_config_defaults(&cfg);
gail_decoder* dec;
if (gail_decoder_create(&cfg, &dec) != GAIL_OK) { puts(gail_last_error()); return 1; }
for (each codestream) {
    gail_packet pkt = {0};
    pkt.data = codestream;
    pkt.size = size;
    gail_image img = {0};
    if (gail_decoder_decode(dec, &pkt, &img) != GAIL_OK) { puts(gail_last_error()); continue; }
    show(img.data[0], img.stride[0], img.width, img.height);  /* xyz12le */
    gail_image_free(&img);
}
gail_decoder_destroy(dec);
```

## Example

[tests/test_api.c](../tests/test_api.c) exercises every mode and format and is the
reference example. The shortest complete use of the encoder:

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
