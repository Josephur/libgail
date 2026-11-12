// SPDX-License-Identifier: MIT
// Bit-level reading and writing of JPEG 2000 packet headers (ISO/IEC 15444-1 B.10.1).
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gail {

// Packet-header bit writer (B.10.1): bits fill each byte MSB first, and a byte that
// follows 0xFF carries only 7 bits (its MSB is the stuffed 0), so no marker code can
// appear in a header. With dst == nullptr it only counts bytes.
class BitWriter {
public:
    explicit BitWriter(std::vector<uint8_t>* dst) : dst_(dst) {}
    void putBit(uint32_t b) {
        acc_ = (acc_ << 1) | (b & 1u);
        if (++used_ == room_) emit(acc_);
    }
    // The n (0..32) low bits of v, MSB first, as many at a time as the current byte takes.
    void putBits(uint32_t v, int n) {
        while (n > 0) {
            const int take = std::min(n, room_ - used_);
            n -= take;
            acc_ = (acc_ << take) | ((v >> n) & ((1u << take) - 1u));
            used_ += take;
            if (used_ == room_) emit(acc_);
        }
    }
    void putZeros(int n) {
        for (; n > 24; n -= 24) putBits(0, 24);
        putBits(0, n);
    }
    // Pads the last byte with zeros. A header ending in 0xFF still owes the stuffed
    // byte that must follow it, so that is written as 0x00. An empty header is one
    // zero byte.
    size_t flush() {
        if (used_ > 0 || count_ == 0) emit(acc_ << (room_ - used_));
        if (last_ == 0xFF) emit(0);
        return count_;
    }

private:
    void emit(uint32_t byte) {
        last_ = byte & 0xFFu;
        if (dst_) dst_->push_back((uint8_t)last_);
        ++count_;
        room_ = last_ == 0xFF ? 7 : 8;
        acc_ = 0;
        used_ = 0;
    }
    std::vector<uint8_t>* dst_;
    uint32_t acc_ = 0, last_ = 0;
    int used_ = 0, room_ = 8;
    size_t count_ = 0;
};

// Packet-header bit reader, the inverse of BitWriter: after a 0xFF byte the next byte
// carries 7 bits. Bits past the end of the data read as zero and set overrun().
class BitReader {
public:
    BitReader(const uint8_t* data, size_t size) : p_(data), size_(size) {}
    uint32_t getBit() {
        if (left_ == 0) load();
        return (cur_ >> --left_) & 1u;
    }
    // n (0..32) bits, MSB first.
    uint32_t getBits(int n) {
        uint32_t v = 0;
        while (n > 0) {
            if (left_ == 0) load();
            const int take = std::min(n, left_);
            left_ -= take;
            n -= take;
            v = (v << take) | ((cur_ >> left_) & ((1u << take) - 1u));
        }
        return v;
    }
    // Ends the header where BitWriter::flush() ended it: the rest of the current byte is
    // padding, a final 0xFF is followed by its stuffed byte, and an empty header is one
    // byte. Returns the number of bytes the header took.
    size_t finish() {
        left_ = 0;
        if (pos_ == 0) load();
        left_ = 0;
        if (last_ == 0xFF) {
            load();
            left_ = 0;
        }
        return pos_;
    }
    bool overrun() const { return overrun_; }

private:
    void load() {
        const bool stuffed = last_ == 0xFF;
        if (pos_ < size_) {
            cur_ = p_[pos_];
        } else {
            cur_ = 0;
            overrun_ = true;
        }
        ++pos_;
        last_ = cur_;
        left_ = stuffed ? 7 : 8;
    }
    const uint8_t* p_;
    size_t size_, pos_ = 0;
    uint32_t cur_ = 0, last_ = 0;
    int left_ = 0;
    bool overrun_ = false;
};

}  // namespace gail
