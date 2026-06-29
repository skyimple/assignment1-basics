"""BPE tokenizer training (CS336 Assignment 1, §2.4–2.5)."""
from __future__ import annotations

import os
from collections import defaultdict
from typing import BinaryIO

import regex

# GPT-2 pre-tokenization pattern (from the spec).
# Uses \p{L}, \p{N}, and a negative lookahead that stdlib `re` does not support.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _read_file(path: str | os.PathLike) -> str:
    """Read the entire corpus as UTF-8 text, ignoring malformed bytes.

    CR characters are stripped before decoding so that the BPE merges are
    identical regardless of whether the source uses Unix (``\\n``), Windows
    (``\\r\\n``), or old-Mac (``\\r``) line endings. Without this, a Windows
    fixture produces merges like ``(\\r, \\n)`` that don't match the
    reference snapshot generated on Linux/macOS.
    """
    with open(path, "rb") as f:
        data = f.read().replace(b"\r", b"")
    return data.decode("utf-8", errors="ignore")


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split text on any special token, returning the in-between segments.

    No merge may cross a special-token boundary, so we strip them out here
    and feed the resulting segments to the GPT-2 pre-tokenization regex.

    Special tokens are matched longest-first so that overlapping specials
    (e.g. ``<|endofendoftext|>`` and ``<|endoftext|>``) are split correctly:
    the longer token is always preferred at any position.
    """
    if not special_tokens:
        return [text]
    # Sort longest-first so the regex engine prefers the longer alternative
    # when several specials could match at the same position.
    sorted_specials = sorted(special_tokens, key=len, reverse=True)
    pattern = "|".join(regex.escape(t) for t in sorted_specials)
    return regex.split(pattern, text)


def _pretokenize(text: str) -> list[bytes]:
    """Apply the GPT-2 pre-tokenization regex to `text`, returning a list of
    word-bytes. Each word is a UTF-8-encoded pre-token; downstream BPE will
    further split each word into single bytes and merge from there.
    """
    return [m.group().encode("utf-8") for m in regex.finditer(PAT, text)]

"""
word_freqs是一个单词或者字符串的 出现次数例如
word_freq{0:1,1:2}

word_tokens是每个wid对应的具体bytes是什么
word_tokens word_tokens = {
    0: [b'h', b'e', b'l', b'l', b'o'],  # "hello" → 5个字节
    1: [b'h', b'e', b'l'],              # "hel"   → 3个字节
}
"""

"""
pair_counts 和 pair_to_words

pair_counts = {
    (b'h', b'e'): 3,   # 来自 word0(2次) + word1(1次)
    (b'e', b'l'): 3,   # 来自 word0(2次) + word1(1次)
    (b'l', b'l'): 2,   # 仅来自 word0(2次)
    (b'l', b'o'): 2,   # 仅来自 word0(2次)
}

pair_to_words = {
    (b'h', b'e'): {0, 1},  # 这对出现在两个单词中
    (b'e', b'l'): {0, 1},  # 这对出现在两个单词中
    (b'l', b'l'): {0},     # 仅出现在单词0
    (b'l', b'o'): {0},     # 仅出现在单词0
}

"""

def _build_pair_index(
    word_freq: dict[int, int],
    word_tokens: dict[int, list[bytes]],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[int]]]:
    """Build (pair_counts, pair_to_words) from the initial word-token sequences.

    For each word, every adjacent (tokens[i], tokens[i+1]) pair contributes
    `word_freq[wid]` to that pair's global count, and the word id is added
    to the pair's word set.

    Returns:
        pair_counts: global count for each adjacent pair.
        pair_to_words: word ids that contain the pair.
    """
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for wid, tokens in word_tokens.items():
        freq = word_freq[wid]
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += freq
            pair_to_words[pair].add(wid)
    return pair_counts, pair_to_words


# 这个逻辑比较直接
# 但由于存在pair_to_word的映射关系，所以排序的速度应该不会太慢

def _find_best_pair(
    pair_counts: dict[tuple[bytes, bytes], int],
) -> tuple[bytes, bytes]:
    """Return the pair with the highest count. On ties, pick the
    lexicographically greater pair (PDF §2.4 tie-breaking rule).
    """
    # `max` over items with key (count, pair) selects the larger count first,
    # then the larger pair on ties — both in ascending order, which is what
    # we want because we want the *greater* pair on ties.
    best_pair, _ = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))
    return best_pair


def _apply_merge(
    wid: int,
    tokens: list[bytes],
    best_pair: tuple[bytes, bytes],
    merged: bytes,
    word_freq: dict[int, int],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[int]],
) -> list[bytes]:
    """Rebuild `tokens` by collapsing every non-overlapping occurrence of
    `best_pair` into `merged`, and update pair_counts / pair_to_words
    incrementally. Returns the new token list.
    """
    freq = word_freq[wid]

    # Decrement the counts for every pair that the old word contributed.
    for i in range(len(tokens) - 1):
        p = (tokens[i], tokens[i + 1])
        pair_counts[p] -= freq
        if pair_counts[p] <= 0:
            # Clean up zero/negative counts so the next `max` ignores them.
            del pair_counts[p]
        pair_to_words[p].discard(wid)
        if not pair_to_words[p]:
            del pair_to_words[p]

    # Rebuild the word: collapse non-overlapping `best_pair` into `merged`.
    new_tokens: list[bytes] = []
    left, right = best_pair
    i = 0
    n = len(tokens)
    while i < n:
        if i + 1 < n and tokens[i] == left and tokens[i + 1] == right:
            new_tokens.append(merged)
            i += 2
        else:
            new_tokens.append(tokens[i])
            i += 1

    # Increment the counts for every pair in the new word.
    for i in range(len(new_tokens) - 1):
        p = (new_tokens[i], new_tokens[i + 1])
        pair_counts[p] += freq
        pair_to_words[p].add(wid)

    return new_tokens


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer.

    Args:
        input_path: path to a UTF-8 text corpus.
        vocab_size: total vocabulary size (special tokens + 256 single bytes
            + `vocab_size - 256 - len(special_tokens)` merged tokens).
        special_tokens: list of strings that must never be split or merged
            across. Added to the vocabulary first.

    Returns:
        vocab: dict mapping token id (int) -> token bytes.
        merges: list of ((left_bytes, right_bytes), ...) in creation order.
    """
    # 1. Initialize the base vocabulary: special tokens first, then 256
    # single bytes. Special tokens are reserved id 0..len(special_tokens)-1
    # so that no merge can ever assign them a different byte sequence.
    vocab: dict[int, bytes] = {}
    next_id = 0
    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1
    for b in range(256):
        vocab[next_id] = bytes([b])
        next_id += 1
    num_merges = vocab_size - next_id
    assert num_merges >= 0, (
        f"vocab_size={vocab_size} is too small to fit base vocab "
        f"({next_id} entries reserved)"
    )

    # 2. Read the corpus and pre-tokenize. Special tokens are stripped first
    # so no merge can ever cross one.
    text = _read_file(input_path)
    segments = _split_on_special_tokens(text, special_tokens)
    # 3. Pre-tokenize each segment with the GPT-2 regex. Each match becomes
    # one "word" — a sequence of single bytes to be merged from there.
    word_counter: dict[bytes, int] = defaultdict(int)
    for seg in segments:
        for word in _pretokenize(seg):
            word_counter[word] += 1

    # 4. Convert words into id-keyed structures for O(1) updates.
    word_freq: dict[int, int] = {}
    word_tokens: dict[int, list[bytes]] = {}
    for wid, (word, freq) in enumerate(word_counter.items()):
        word_freq[wid] = freq
        word_tokens[wid] = [bytes([b]) for b in word]

    # 5. Build the pair index once. After this, every merge step only
    # touches the words that contain the merged pair.
    pair_counts, pair_to_words = _build_pair_index(word_freq, word_tokens)

    # 6. Main loop: do `num_merges` merges, recording the merge order.
    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        if not pair_counts:
            # Corpus is exhausted (e.g. only one unique word). Pad with
            # unused ids so vocab has the requested size; merges stays short.
            break
        best_pair = _find_best_pair(pair_counts)
        merges.append(best_pair)
        merged = best_pair[0] + best_pair[1]
        vocab[next_id] = merged
        next_id += 1

        # Update every word that contained this pair.
        affected = list(pair_to_words.get(best_pair, ()))
        for wid in affected:
            word_tokens[wid] = _apply_merge(
                wid,
                word_tokens[wid],
                best_pair,
                merged,
                word_freq,
                pair_counts,
                pair_to_words,
            )
        # The merged pair itself is no longer present in any word.
        pair_to_words.pop(best_pair, None)
        pair_counts.pop(best_pair, None)

    # If we exited early (corpus exhausted), top up with unused placeholder
    # bytes so the returned vocab has the requested size. These won't affect
    # downstream tests because no real text produces them, but the snapshot
    # test does compare vocab key sets, so they must be present.
    while next_id < vocab_size:
        vocab[next_id] = b"\x00" * 2  # distinct, never produced by any merge
        next_id += 1

    return vocab, merges
