# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent role — read first

This is a **student assignment** repo (CS336 Spring 2025 Assignment 1). The project's `AGENTS.md` defines agent behavior: agents are **teaching aids**, not solution generators. When a student asks for help, prefer explanation, conceptual guidance, debugging questions, and review of code the student wrote. Do not write the assignment solution. The repo already contains a `train_bpe.py` and `tokenizer.py` (these are the student's own files); future agents should be aware these are not solutions to preserve or polish — they may be incomplete and the student may ask for review.

## Commands

All Python commands go through `uv` (Python 3.12–3.13, see `pyproject.toml`).

```sh
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_model.py

# Run a single test
uv run pytest tests/test_model.py -k test_linear

# Grading command: 10s timeout per test, JUnit XML output (used by make_submission.sh)
uv run pytest --timeout 10 -v ./tests --junitxml=test_results.xml

# Snapshot tests: stricter tolerance (atol=rtol=0)
uv run pytest --snapshot-exact

# Lint
uv run ruff check .

# Run any Python file in the package
uv run python cs336_basics/pretokenization_example.py

# Build the submission zip (CI / grading)
./make_submission.sh
```

`pyproject.toml` sets `addopts = "-s"` for pytest (no capture) and `log_cli = true`, so test output streams directly.

## Architecture

### Assignment structure

Students build a Transformer LM from scratch in PyTorch. The repo ships tests but no scaffolding except `tests/adapters.py`. The student's own code lives in `cs336_basics/` (currently `train_bpe.py`, `tokenizer.py`, `pretokenization_example.py`).

### The adapter pattern — `tests/adapters.py`

The test suite calls functions in `tests/adapters.py`, never student modules directly. Each adapter is a thin shim:

- Already wired: `run_train_bpe` → `cs336_basics.train_bpe.train_bpe`; `get_tokenizer` → `cs336_basics.tokenizer.Tokenizer`.
- Still stubs (`raise NotImplementedError`): the model/optimizer/nn-utility adapters in the same file — these will fail until the student implements them.

Adapters should call through to student classes/functions and are the **only** thing the tests import from the student side.

### Snapshot testing — `tests/conftest.py`

Two fixtures:

- `snapshot` — pickles arbitrary Python objects; expected outputs live in `tests/_snapshots/*.pkl`. Used by `test_train_bpe_special_tokens`.
- `numpy_snapshot` — `.npz` files via `numpy.testing.assert_allclose` with default `rtol=1e-4, atol=1e-2`. Used by the model tests. Honors `--snapshot-exact` (forces atol=rtol=0).

`default_test_name` is set to `request.node.name`, so snapshots are keyed by test function name. Other fixtures (`q`, `k`, `v`, `in_embeddings`, `mask`, `in_indices`, `pos_ids`, `theta`) seed torch/numpy for deterministic component tests; `ts_state_dict` loads the pretrained TinyStories checkpoint from `tests/fixtures/ts_tests/`.

### Shared utilities — `tests/common.py`

`FIXTURES_PATH` (path to `tests/fixtures/`) and `gpt2_bytes_to_unicode()` — the byte↔unicode remap used by GPT-2's vocab/merges. The test fixture `get_tokenizer_from_vocab_merges_path` in `tests/test_tokenizer.py` inverts this map and feeds raw bytes to `get_tokenizer`.

### Test modules

| File | Covers |
|---|---|
| `test_nn_utils.py` | softmax, cross-entropy, gradient clipping |
| `test_model.py` | Linear, Embedding, SiLU, SwiGLU, RMSNorm, SDPA, MHA, RoPE, block, full LM |
| `test_optimizer.py` | AdamW, cosine LR + linear warmup |
| `test_data.py` | `get_batch` (random contiguous sequences with x/y offset by 1) |
| `test_tokenizer.py` | `Tokenizer` encode/decode/streaming, special tokens, tiktoken parity |
| `test_train_bpe.py` | BPE training: correctness vs reference merges/vocab, speed < 1.5s, special-token snapshot |
| `test_serialization.py` | checkpoint save/load roundtrip |

### BPE architecture notes

- The GPT-2 pre-tokenization regex `PAT` in `cs336_basics/train_bpe.py` requires the third-party `regex` package (stdlib `re` doesn't support `\p{L}` / `\p{N}` / negative lookaheads).
- `_split_on_special_tokens` matches specials **longest-first** so overlapping specials like `<|endofendoftext|>` vs `<|endoftext|>` don't get broken into two tokens. Same module exposes `_pretokenize`, `_build_pair_index`, `_find_best_pair` (highest count, ties → lexicographically greater pair per PDF §2.4), and `_apply_merge`.
- Tie-break on merge selection uses `max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))` — `max` returns greater element on ties because Python's ordering is ascending.

### Platform quirk — Windows

`tests/test_tokenizer.py` does a bare `import resource` at module top, which fails on Windows (`ModuleNotFoundError: No module named 'resource'`). The `test_encode_iterable_memory_usage` and `test_encode_memory_usage` tests use `resource.setrlimit(RLIMIT_AS, ...)` and are intentionally Linux-only (guarded by `@pytest.mark.skipif(not linux)`).

On Windows, full module collection of `test_tokenizer.py` fails before any test runs. The same file also opens `gpt2_merges.txt` without `encoding='utf-8'`, which fails on Windows because the default encoding is cp936/cp1252 and the file contains `Ġ` (U+0120). On Linux both work transparently.

## External resources

- Assignment handout: `cs336_assignment1_basics.pdf` (Typst source; current version 26.0.3 per `CHANGELOG.md`).
- Lecture materials: https://cs336.stanford.edu (per `AGENTS.md`).
- TinyStories + OpenWebText data download commands are in `README.md`.
- GPT-2 reference vocab/merges used by the tokenizer tests live in `tests/fixtures/`.