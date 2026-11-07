"""Integer coding for the index files: variable-byte numbers and bit-packed blocks.

* ``varint``: 7 bits per byte, the high bit says "more bytes follow" (the format of protobuf and LEB128; 300 is ``AC 02``).
* ``pack_block`` / ``unpack_block``: frame-of-reference coding.  A block of up to ``BLOCK`` numbers is written with the width in bits of
  its largest number, one byte for the width and then the numbers back to back.  Small gaps between document ids and small term
  frequencies need only a few bits each.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

BLOCK = 128


def write_varint(out: bytearray, value: int) -> None:
    if value < 0:
        raise ValueError("varint cannot hold a negative number")
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)


def encode_varint(value: int) -> bytes:
    out = bytearray()
    write_varint(out, value)
    return bytes(out)


def read_varint(buf, pos: int) -> Tuple[int, int]:
    """(value, position after it)."""
    b = buf[pos]
    if b < 0x80:
        return b, pos + 1
    value = b & 0x7F
    shift = 7
    pos += 1
    while True:
        b = buf[pos]
        pos += 1
        value |= (b & 0x7F) << shift
        if b < 0x80:
            return value, pos
        shift += 7


def read_varints(buf, pos: int, count: int) -> Tuple[List[int], int]:
    out = []
    append = out.append
    for _ in range(count):
        b = buf[pos]
        pos += 1
        if b < 0x80:
            append(b)
            continue
        value = b & 0x7F
        shift = 7
        while True:
            b = buf[pos]
            pos += 1
            value |= (b & 0x7F) << shift
            if b < 0x80:
                break
            shift += 7
        append(value)
    return out, pos


def bit_width(value: int) -> int:
    return value.bit_length()


def pack_block(values: Sequence[int]) -> bytes:
    """One byte for the width, then ``len(values) * width`` bits (the caller knows how many numbers there are)."""
    width = max(values).bit_length() if values else 0
    if width == 0:
        return b"\x00"
    acc = 0
    shift = 0
    for v in values:
        acc |= v << shift
        shift += width
    return bytes([width]) + acc.to_bytes((shift + 7) // 8, "little")


def packed_size(count: int, width: int) -> int:
    return 1 + (count * width + 7) // 8 if width else 1


def unpack_block(buf, pos: int, count: int) -> Tuple[List[int], int]:
    """(numbers, position after the block)."""
    width = buf[pos]
    if width == 0:
        return [0] * count, pos + 1
    nbytes = (count * width + 7) // 8
    acc = int.from_bytes(buf[pos + 1 : pos + 1 + nbytes], "little")
    mask = (1 << width) - 1
    if width == 8:
        return list(buf[pos + 1 : pos + 1 + count]), pos + 1 + nbytes
    return [(acc >> (i * width)) & mask for i in range(count)], pos + 1 + nbytes
