#!/bin/sh
# SPDX-License-Identifier: MIT
# Adds the libgail encoder and decoder to an FFmpeg source tree (tested with FFmpeg 9.0):
#
#   integrations/ffmpeg/add-to-ffmpeg.sh /path/to/ffmpeg
#   cd /path/to/ffmpeg && ./configure --enable-libgail ... && make
#
# configure finds the library with pkg-config ("gail"); set PKG_CONFIG_PATH if Gail is
# installed somewhere unusual. It needs a libgail with the decoder (1.1.0 or later):
# configure checks for gail_decoder_create, which 1.0.x does not have. Running this twice
# is harmless.
set -e
here=$(cd "$(dirname "$0")" && pwd)
ff=${1:?usage: add-to-ffmpeg.sh FFMPEG_SOURCE_DIR}
cd "$ff"
test -f libavcodec/allcodecs.c || { echo "not an FFmpeg source tree: $ff" >&2; exit 1; }

cp "$here/libgailenc.c" libavcodec/libgailenc.c
cp "$here/libgaildec.c" libavcodec/libgaildec.c

grep -q ff_libgail_encoder libavcodec/allcodecs.c ||
    sed -i 's/^extern const FFCodec ff_libopenjpeg_encoder;/extern const FFCodec ff_libgail_encoder;\n&/' libavcodec/allcodecs.c
grep -q ff_libgail_decoder libavcodec/allcodecs.c ||
    sed -i 's/^extern const FFCodec ff_libgail_encoder;/&\nextern const FFCodec ff_libgail_decoder;/' libavcodec/allcodecs.c

grep -q LIBGAIL_ENCODER libavcodec/Makefile ||
    sed -i 's/^OBJS-$(CONFIG_LIBOPENJPEG_ENCODER) .*/OBJS-$(CONFIG_LIBGAIL_ENCODER)            += libgailenc.o\n&/' libavcodec/Makefile
grep -q LIBGAIL_DECODER libavcodec/Makefile ||
    sed -i 's/^OBJS-$(CONFIG_LIBGAIL_ENCODER) .*/OBJS-$(CONFIG_LIBGAIL_DECODER)            += libgaildec.o\n&/' libavcodec/Makefile

if ! grep -q enable-libgail configure; then
    sed -i 's/^  --enable-libopenjpeg     enable JPEG 2000 encoding via OpenJPEG \[no\]/  --enable-libgail         enable DCI JPEG 2000 encoding and decoding on the GPU via Gail [no]\n&/' configure
    sed -i 's/^    libopenjpeg$/    libgail\n&/' configure
    sed -i 's/^libopenjpeg_encoder_deps="libopenjpeg"/libgail_encoder_deps="libgail"\n&/' configure
    sed -i 's/^enabled libopenjpeg       && /enabled libgail           \&\& require_pkg_config libgail "gail >= 1.0.1" gail\/gail.h gail_decoder_create\n&/' configure
fi
# a tree that already had the encoder gains the decoder, and the check for a libgail that has it
grep -q libgail_decoder_deps configure ||
    sed -i 's/^libgail_encoder_deps="libgail"/libgail_decoder_deps="libgail"\n&/' configure
sed -i 's/require_pkg_config libgail "gail >= 1\.0\.0" gail\/gail\.h gail_encoder_create/require_pkg_config libgail "gail >= 1.0.1" gail\/gail.h gail_decoder_create/' configure
grep -q 'gail_decoder_create' configure || { echo "failed to patch configure's libgail check" >&2; exit 1; }

for f in libavcodec/allcodecs.c libavcodec/Makefile configure; do
    grep -qi libgail "$f" || { echo "failed to patch $f" >&2; exit 1; }
done
echo "libgail added to $ff"
