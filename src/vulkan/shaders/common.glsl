// SPDX-License-Identifier: MIT
// Shared declarations for Gail compute shaders (buffer device addresses).
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require

layout(buffer_reference, std430, buffer_reference_align = 4) buffer U32Buf { uint v[]; };
layout(buffer_reference, std430, buffer_reference_align = 16) buffer U32x4Buf { uvec4 v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer I32Buf { int v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer F32Buf { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 2) buffer U16Buf { uint16_t v[]; };
layout(buffer_reference, std430, buffer_reference_align = 1) buffer U8Buf { uint8_t v[]; };

// Code-block table entry (4 words): band, absolute x, absolute y, w | h << 8 | orient << 16 | comp << 20
#define CB_W(p) int((p) & 0xFFu)
#define CB_H(p) int(((p) >> 8) & 0xFFu)
#define CB_ORIENT(p) int(((p) >> 16) & 0xFu)
#define CB_COMP(p) uint(((p) >> 20) & 0xFu)

#define MAX_PASSES 64
#define MAX_PLANES 24
