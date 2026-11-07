import random

import pytest

from kendi_arama.codec import BLOCK, encode_varint, pack_block, read_varint, read_varints, unpack_block, write_varint


def test_varint_known_values():
    # the examples of the protobuf documentation
    assert encode_varint(1) == b"\x01"
    assert encode_varint(127) == b"\x7f"
    assert encode_varint(128) == b"\x80\x01"
    assert encode_varint(300) == b"\xac\x02"
    assert encode_varint(0) == b"\x00"
    assert encode_varint(2**32) == b"\x80\x80\x80\x80\x10"


def test_varint_roundtrip():
    rng = random.Random(1)
    values = [0, 1, 127, 128, 16383, 16384, 2**21 - 1, 2**21, 2**35, 2**63 + 5] + [rng.getrandbits(rng.randint(1, 60)) for _ in range(500)]
    buf = bytearray()
    for v in values:
        write_varint(buf, v)
    pos = 0
    for v in values:
        got, pos = read_varint(bytes(buf), pos)
        assert got == v
    assert pos == len(buf)
    again, pos = read_varints(bytes(buf), 0, len(values))
    assert again == values and pos == len(buf)


def test_varint_rejects_negative():
    with pytest.raises(ValueError):
        encode_varint(-1)


@pytest.mark.parametrize("width", list(range(0, 33)) + [40, 63])
def test_pack_block_roundtrip_every_width(width):
    rng = random.Random(width)
    for count in (1, 2, 7, 8, 9, 127, BLOCK):
        values = [rng.getrandbits(width) if width else 0 for _ in range(count)]
        if width:
            values[rng.randrange(count)] = (1 << width) - 1  # the largest value fixes the width
        packed = pack_block(values)
        assert packed[0] == (max(values).bit_length())
        got, pos = unpack_block(packed + b"\xff\xff", 0, count)
        assert got == values and pos == len(packed)


def test_pack_block_is_small_for_small_numbers():
    assert len(pack_block([1] * BLOCK)) == 1 + BLOCK // 8  # one bit each
    assert len(pack_block([0] * BLOCK)) == 1
    assert len(pack_block(list(range(BLOCK)))) == 1 + BLOCK * 7 // 8


def test_unpack_at_an_offset():
    a = pack_block([5, 6, 7])
    b = pack_block([1000, 2000])
    buf = b"xx" + a + b
    first, pos = unpack_block(buf, 2, 3)
    second, end = unpack_block(buf, pos, 2)
    assert first == [5, 6, 7] and second == [1000, 2000] and end == len(buf)
