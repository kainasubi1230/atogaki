from __future__ import annotations

import hashlib


def stable_int_token(value: str, mod: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % max(1, mod)


def katakana_to_hiragana(char: str) -> str:
    if len(char) != 1:
        return char
    code = ord(char)
    if 0x30A1 <= code <= 0x30F6:
        return chr(code - 0x60)
    return char


def normalize_char_for_model(char: str) -> str:
    return katakana_to_hiragana(char)


def char_to_model_id(char: str, vocab_size: int = 4096) -> int:
    normalized = normalize_char_for_model(char)
    return stable_int_token(normalized, vocab_size)

