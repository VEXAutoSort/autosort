"""The jog command's key decoding (arrow keys arrive as escape sequences)."""
import os

import pytest

from machine import read_key


def _key(data: bytes) -> str:
    r, w = os.pipe()
    os.write(w, data)
    os.close(w)
    try:
        return read_key(r)
    finally:
        os.close(r)


@pytest.mark.parametrize("data, expected", [
    (b"\x1b[C", "right"), (b"\x1b[D", "left"), (b"\x1b[A", "up"), (b"\x1b[B", "down"),
    (b" ", "space"), (b"q", "q"), (b"t", "t"), (b"[", "["), (b"]", "]"), (b"h", "h"),
    (b"\x03", "q"),            # Ctrl-C in cbreak mode
    (b"\x1b[5~", ""),          # PageUp: an escape sequence we don't use is swallowed
    (b"x", ""),                # unknown key ignored
])
def test_read_key(data, expected):
    assert _key(data) == expected


def test_read_key_times_out_with_no_input():
    r, w = os.pipe()
    try:
        assert read_key(r) == ""
    finally:
        os.close(r); os.close(w)
