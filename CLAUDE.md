# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Agent role — read first

This is a **student assignment** repo (CS336 Spring 2025 Assignment 1). `AGENTS.md` is the source of truth for agent behavior — read it before doing anything. In short: agents are **teaching aids, not solution generators**. Do not write the assignment solution, do not paste working implementations of core components (tokenizers, transformer blocks, optimizers, training loops, …). Prefer explanation, guiding questions, and review of code the student wrote. When in doubt, refuse the direct implementation and pivot to explanation, debugging guidance, or a high-level outline.

The repo already contains student-written code in `cs336_basics/`. It is **not** a polished solution to preserve — it may be incomplete, and the student may ask for review.

## Commands

All Python commands go through `uv` (Python 3.12–3.13, see `pyproject.toml`). Data download commands are in `README.md`.

```sh
# Run a single test (preferred — see "Windows caveat" below)
uv run pytest tests/test_model.py::test_linear

# Run a whole test file
uv run pytest tests/test_model.py

# Grading command (CI / make_submission.sh): 10s timeout per test, JUnit XML output
uv run pytest --timeout 10 -v ./tests --junitxml=test_results.xml

# Stricter snapshot tolerance (forces atol=rtol=0)
uv run pytest --snapshot-exact

# Lint
uv run ruff check .

# Run any Python file in the package
uv run python cs336_basics/<file>.py

# Build the submission zip
./make_submission.sh
```

`pyproject.toml` sets `addopts = "-s"` (no capture) and `log_cli = true`, so test output streams directly.

## Architecture

Students build a Transformer LM from scratch in PyTorch. The repo ships tests but **no scaffolding** — the only thing the test suite imports from the student side is `tests/adapters.py`.

### The adapter pattern

Every test goes through a thin shim in `tests/adapters.py`. The shim constructs / loads the student's class, then calls it. Each `run_*` adapter maps to a `Problem` deliverable in the PDF (`cs336_assignment1_basics.pdf`).

Currently wired adapters: `run_train_bpe` → `cs336_basics.train_bpe.train_bpe`, `get_tokenizer` → `cs336_basics.tokenizer.Tokenizer`, `run_linear` → `cs336_basics.nn.Linear`.

Still `raise NotImplementedError` (will fail until the student implements the corresponding `Problem` in the PDF): `run_embedding`, `run_silu`, `run_swiglu`, `run_rmsnorm`, `run_scaled_dot_product_attention`, `run_multihead_self_attention` (+ RoPE variant), `run_rope`, `run_transformer_block`, `run_transformer_lm`, `run_softmax`, `run_cross_entropy`, `run_gradient_clipping`, `run_get_adamw_cls`, `run_get_lr_cosine_schedule`, `run_get_batch`, `run_save_checkpoint`, `run_load_checkpoint`.

Student code lives in `cs336_basics/` — currently `train_bpe.py`, `tokenizer.py`, `pretokenization_example.py`, and `nn.py` (the new home for `Linear` and future modules like `Embedding`, `RMSNorm`, `SwiGLU`, `TransformerBlock`, `TransformerLM`).

### Snapshot testing

`tests/conftest.py` provides:
- `numpy_snapshot` — `.npz` files via `numpy.testing.assert_allclose` with default `rtol=1e-4, atol=1e-2`. Honors `--snapshot-exact` (forces `atol=rtol=0`). Used by the model tests.
- `snapshot` — pickles arbitrary Python objects; expected outputs in `tests/_snapshots/*.pkl`. Used by `test_train_bpe_special_tokens`.

`default_test_name = request.node.name`, so snapshots are keyed by test function name. Deterministic component tests use fixtures (`q`, `k`, `v`, `in_embeddings`, `mask`, `in_indices`, `pos_ids`, `theta`) that seed torch/numpy; `ts_state_dict` loads the pretrained TinyStories checkpoint from `tests/fixtures/ts_tests/`.

### §3.3.1 initialization scheme (assignment-wide)

The PDF prescribes one truncated-normal scheme shared across modules — reuse it instead of re-deriving:

- **Linear weights**: `N(0, 2 / (in_features + out_features))` truncated to `[-3, 3]` → `nn.init.trunc_normal_(w, mean=0, std=sqrt(2/(in+out)), a=-3, b=3)`
- **Embedding**: `N(0, 1)` truncated to `[-3, 3]`
- **RMSNorm**: gain parameter only

### §3.3.2 `Linear` conventions (non-obvious — read before adding adjacent modules)

- Signature mirrors `torch.nn.Linear` minus bias: `Linear(in_features, out_features, device=None, dtype=None)`.
- Store the parameter as `W` of shape `(out_features, in_features)`, **not** as `W.T` — the PDF is explicit.
- The adapter is expected to load weights via `layer.load_state_dict({"weight": weights})`, not direct `.data = …`.
- Forward: `x @ self.weight.T` — broadcasts over arbitrary leading batch dims.
- The student reference implementation is in `cs336_basics/nn.py`. Future nn modules go in the same file using the same conventions (same signature, same init helper, same `nn.Module` subclass style).

### BPE architecture notes (`cs336_basics/train_bpe.py`)

- The GPT-2 pre-tokenization regex `PAT` requires the third-party `regex` package (stdlib `re` doesn't support `\p{L}` / `\p{N}` / negative lookaheads).
- `_split_on_special_tokens` matches specials **longest-first** so overlapping specials like `<|endofendoftext|>` vs `<|endoftext|>` don't get split into two tokens.
- `_find_best_pair` uses `max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))` — `max` returns the greater element on ties because Python's ordering is ascending (PDF §2.4).
- Same module exposes `_pretokenize`, `_build_pair_index`, and `_apply_merge`.

## Platform quirks

### Windows — `test_tokenizer.py` breaks collection

`tests/test_tokenizer.py` does a bare `import resource` at module top, which raises `ModuleNotImplementedError` on Windows (`resource` is POSIX-only). The two memory tests using `resource.setrlimit` are correctly `@pytest.mark.skipif(not linux)`, but the bare import happens **before** that skip is evaluated.

Consequence: `uv run pytest` (full collection) and `uv run pytest -k <anything>` both crash at collection. Run tests by **file** instead (`uv run pytest tests/test_model.py::test_linear`), which avoids importing `test_tokenizer.py`.

The same file also opens `gpt2_merges.txt` without `encoding='utf-8'`, which fails on Windows because the default encoding is cp936/cp1252 and the file contains `Ġ` (U+0120).

### Windows — pushing to GitHub over SSH

This machine's outbound network blocks TCP/22 to `github.com`. `git push` over the existing SSH remote hangs and times out. The repo and credentials are fine; switching to a different network makes the push succeed. Fallbacks if the network can't be changed: `git remote set-url origin https://github.com/skyimple/assignment1-basics.git`, or rewrite `~/.ssh/config` to use port 443 via `ssh.github.com`.

## External resources

- Assignment handout: `cs336_assignment1_basics.pdf` (Typst source; current version 26.0.3 per `CHANGELOG.md`).
- Lecture materials: https://cs336.stanford.edu (per `AGENTS.md`).
- TinyStories + OpenWebText download commands: `README.md`.
- GPT-2 reference vocab/merges for tokenizer tests: `tests/fixtures/`.