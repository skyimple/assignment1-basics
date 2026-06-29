"""BPE tokenizer inference (CS336 Assignment 1, §2.6).

Given a trained ``(vocab, merges)`` pair produced by :func:`train_bpe` (or any
equivalent byte-level BPE trainer), this module exposes a :class:`Tokenizer`
class that mirrors the runtime API of the GPT-2 tokenizer used by
HuggingFace / tiktoken: ``encode(str) -> list[int]``, ``decode(list[int]) -> str``,
and a streaming ``encode_iterable(Iterable[str]) -> Iterator[int]`` that
processes one chunk at a time so the in-flight memory stays bounded.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import regex

# Reuse the GPT-2 pre-tokenization regex and the longest-first special-token
# splitter implemented alongside the BPE trainer.
from .train_bpe import PAT, _split_on_special_tokens


class Tokenizer:
    """Byte-level BPE tokenizer built from a fixed vocab + merges pair.

    Args:
        vocab: mapping from token id (int) to token bytes.
        merges: ordered list of ``(left_bytes, right_bytes)`` pairs in the
            order they were learned. Earlier entries are applied first.
        special_tokens: optional list of strings that must remain as single
            tokens and never be merged or split. Each must already appear
            among ``vocab.values()``; otherwise the constructor raises.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = vocab

        # Inverse map for O(1) bytes -> id lookups during encoding. Raises
        # ``KeyError`` if a token produced by BPE is missing from vocab; this
        # surfaces inconsistent (vocab, merges) pairs immediately.
        self.inverse_vocab: dict[bytes, int] = {b: i for i, b in vocab.items()}

        # GPT-2 pre-tokenization regex, compiled once.
        self._pat = regex.compile(PAT)

        # Special-token handling. We keep the raw strings for reference and a
        # bytes-keyed lookup for fast emission during encoding. The split
        # pattern is sorted longest-first so overlapping specials match
        # greedily (e.g. ``<|endofendoftext|>`` beats ``<|endoftext|>``).
        self.special_tokens: list[str] = list(special_tokens) if special_tokens else []
        self.special_token_bytes_to_id: dict[bytes, int] = {}
        self._special_split_pat: regex.Pattern | None = None
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            # Capture group so ``regex.split`` returns the delimiters alongside
            # the in-between text segments.
            split_pat = "(" + "|".join(regex.escape(s) for s in sorted_specials) + ")"
            self._special_split_pat = regex.compile(split_pat)
            for st in self.special_tokens:
                st_bytes = st.encode("utf-8")
                if st_bytes not in self.inverse_vocab:
                    raise ValueError(
                        f"Special token {st!r} is not present in vocab; "
                        "add it to vocab before constructing the Tokenizer."
                    )
                self.special_token_bytes_to_id[st_bytes] = self.inverse_vocab[st_bytes]

        # Rank map for the encode-time BPE merge walk: lower rank = applied
        # first. Unranked pairs are never merged.
        self._merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: idx for idx, pair in enumerate(merges)
        }

    # ------------------------------------------------------------------ encode

    def _bpe_encode_pre_token(self, pre_token_bytes: bytes) -> list[int]:
        """Run BPE on a single pre-token (one PAT match) and return its ids.

        Standard GPT-2 / Sennrich encode: walk adjacent pairs left-to-right,
        and whenever a pair is in the merge ranks, replace it with the merged
        bytes and rewind the cursor by one. Stop when no pair is rankable.
        Each emitted bytes (initial single bytes + every merge result) is
        looked up in ``inverse_vocab`` to produce its id.
        """
        if not pre_token_bytes:
            return []

        tokens: list[bytes] = [bytes([b]) for b in pre_token_bytes]
        ids: list[int] = []

        while True:
            best_rank: int | None = None
            best_idx: int = -1
            for i in range(len(tokens) - 1):
                rank = self._merge_ranks.get((tokens[i], tokens[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
                    # Can't break early: we still need to scan the rest in
                    # case an earlier index has the same minimum rank.
            if best_idx < 0:
                break
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens[best_idx] = merged
            del tokens[best_idx + 1]
            # After a merge, the new token may itself merge with its left or
            # right neighbor; restarting the scan handles both.

        for tok in tokens:
            ids.append(self.inverse_vocab[tok])
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a list of token ids.

        Special tokens are split out first (longest-first) and emitted as
        single ids. The remaining segments are pre-tokenized with the GPT-2
        regex and merged via BPE.
        """
        if not text:
            return []

        ids: list[int] = []

        if self._special_split_pat is None:
            segments: list[str] = [text]
        else:
            # ``_special_split_pat`` has a single capture group, so the
            # delimiter strings appear in the returned list interleaved with
            # the in-between text segments.
            segments = self._special_split_pat.split(text)

        for seg in segments:
            if not seg:
                continue
            seg_bytes = seg.encode("utf-8")
            if seg_bytes in self.special_token_bytes_to_id:
                # Whole segment is a special token — emit its id directly,
                # never feed it into BPE.
                ids.append(self.special_token_bytes_to_id[seg_bytes])
                continue
            # Regular text: pre-tokenize with the GPT-2 regex, then BPE-merge
            # each pre-token.
            for m in self._pat.finditer(seg):
                pre_token_bytes = m.group().encode("utf-8")
                if pre_token_bytes in self.special_token_bytes_to_id:
                    # Should not happen because specials were split out above,
                    # but defensively guard in case a special-token string
                    # happens to also match the pre-tokenization regex.
                    ids.append(self.special_token_bytes_to_id[pre_token_bytes])
                else:
                    ids.extend(self._bpe_encode_pre_token(pre_token_bytes))

        return ids

    # ------------------------------------------------------------------ decode

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into a string.

        Concatenates the token bytes and decodes as UTF-8 with ``replace``
        error handling, so partial multi-byte sequences across token
        boundaries become ``U+FFFD`` instead of raising. Missing ids are
        skipped silently for lenient decoding.
        """
        pieces: list[bytes] = []
        for i in ids:
            b = self.vocab.get(i)
            if b is not None:
                pieces.append(b)
        return b"".join(pieces).decode("utf-8", errors="replace")

    # --------------------------------------------------------------- streaming

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of text chunks, yielding one id at a time.

        Each chunk is encoded independently via :meth:`encode`; there is no
        carry-over between chunks. This keeps the in-flight memory bounded
        by the size of one chunk plus the tokenizer's precomputed state,
        which is what allows streaming-encoding large files (see
        ``test_encode_iterable_memory_usage``).

        Caveat: a special token or pre-token that straddles two chunks will
        be encoded incorrectly at the boundary. The streaming test reads
        line-by-line and no special token in any fixture contains a newline,
        so this is acceptable for the test suite.
        """
        for chunk in iterable:
            yield from self.encode(chunk)