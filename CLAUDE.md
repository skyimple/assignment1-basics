# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```sh
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_model.py

# Run a single test
uv run pytest tests/test_model.py -k test_linear

# Run all tests with 10s timeout (used for grading submission)
uv run pytest --timeout 10 -v ./tests --junitxml=test_results.xml

# Run with exact snapshot matching (stricter tolerances)
uv run pytest --snapshot-exact

# Lint check
uv run ruff check .

# Run any Python file
uv run python cs336_basics/pretokenization_example.py
```

## Architecture

### Assignment structure

This is CS336 Assignment 1 — students build a Transformer language model from scratch in PyTorch. The repo provides tests but no implementation scaffolding (except `tests/adapters.py`). Students create their own module files under `cs336_basics/`.

### Test infrastructure

**`tests/adapters.py`** — The bridge between student code and the test suite. Every function here raises `NotImplementedError` initially. Students implement these adapter functions (e.g., `run_linear(…)`, `run_transformer_lm(…)`) to connect their own PyTorch modules to the tests. Adapters should call through to the student's own classes/functions.

**`tests/conftest.py`** — Shared pytest fixtures:
- `numpy_snapshot` / `snapshot` — Snapshot comparators that assert output matches stored `.npz`/`.pkl` files in `tests/_snapshots/`
- `ts_state_dict` — Loads a pretrained TinyStories model checkpoint and config from `tests/fixtures/ts_tests/`
- Tensor fixtures (`q`, `k`, `v`, `in_embeddings`, `mask`, `in_indices`, `pos_ids`) — Seeded random tensors for model component tests

**`tests/common.py`** — Shared utility: `gpt2_bytes_to_unicode()` mapping for BPE tokenizer work.

**`tests/_snapshots/`** — Expected outputs for snapshot-based tests (`.npz` for numeric, `.pkl` for data). Tests compare student output against these at `rtol=1e-4, atol=1e-2`.

**`tests/fixtures/`** — Test data: GPT-2 vocab/merges files, TinyStories samples, BPE training corpus, transformer model checkpoints.

### Test modules

| File | What it tests |
|---|---|
| `test_nn_utils.py` | Softmax, cross-entropy loss, gradient clipping |
| `test_model.py` | Linear, embedding, SiLU, SwiGLU, RMSNorm, SDPA, multi-head attention, RoPE, transformer block, full transformer LM |
| `test_optimizer.py` | AdamW optimizer, cosine LR schedule with linear warmup |
| `test_data.py` | Batch sampling (random contiguous sequences, x/y offset by 1) |
| `test_tokenizer.py` | BPE tokenizer construction, encode/decode, memory efficiency |
| `test_train_bpe.py` | BPE training (correctness against reference merges/vocab, speed < 1.5s) |
| `test_serialization.py` | Checkpoint save/load roundtrip |

### Key tooling

- **Package manager**: `uv` — `uv run <command>` for anything Python-related
- **Python**: 3.12–3.13
- **Linter**: `ruff` (line-length 120)
- **Snapshot tests**: Outputs match stored `.npz` snapshots via `numpy.testing.assert_allclose`
- **Type hints**: `jaxtyping` (e.g., `Float[Tensor, "batch d_model"]`) used throughout adapters
