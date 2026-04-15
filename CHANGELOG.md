# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-04-15

Breaking release. Sub-sweep dimensionality is no longer hardcoded to 2.

### Changed (BREAKING)
- `run_optimizer` now accepts `fine_steps: dict[str, float]` and
  `fine_margins: dict[str, float]` keyed by axis name. The old
  positional kwargs `fine_step_0, fine_step_1, fine_margin_0,
  fine_margin_1` are **removed**. Sub-sweep dimensionality follows
  `len(problem.axis_names())`, so Problems can now expose 1, 2, 3, …
  search axes.
- Consumers pinned at `evolutionary-solver>=0.1,<0.2` must bump to
  `>=0.2,<0.3` and update their call sites to pass dicts. No in-code
  shim is provided; the pin IS the migration barrier.

### Added
- N-axis sub-sweep support throughout `run_optimizer`: grid
  construction, basin extraction, fine-zoom centering, progress
  output all iterate over `problem.axis_names()`.

### Fixed
- Basin `min_separation` now takes the conservative (smallest)
  per-axis ratio rather than only axis 0 — matches the Chebyshev
  distance check inside `extract_basins` across all dimensions.
- `_chebyshev_shell` rewritten to enumerate the shell surface
  directly instead of iterating the full `(2r+1)^ndim` cube and
  filtering; constant-factor speedup (2–3× in 2-D, shrinking with
  N). Docstring notes that proper high-N seeding needs a different
  algorithm (KD-tree / Poisson-disk) before extending to ≥5 axes.

### Migration
- State-file format is unchanged: `optimizer_state.json` still
  stores per-axis keys `fine_step_{axis_name}` and
  `fine_margin_{axis_name}`. 2-axis runs saved under 0.1.x resume
  cleanly under 0.2.0.
- Call-site migration (per caller):
  ```python
  # 0.1.x
  run_optimizer(..., fine_step_0=0.5, fine_step_1=0.25,
                fine_margin_0=2.0, fine_margin_1=1.0, ...)
  # 0.2.0
  run_optimizer(..., fine_steps={"ax0": 0.5, "ax1": 0.25},
                fine_margins={"ax0": 2.0, "ax1": 1.0}, ...)
  ```

## [0.1.1] — 2026-04-13

### Added
- `run_optimizer(seed_candidates=...)` — optional list of known-good
  genomes injected as the first N members of gen0 ahead of LHS random
  candidates. Remaining `pop_size − N` candidates are sampled as before
  and renumbered to follow the seeds.

### Fixed
- `n_remaining` progress counter went negative when `max_attempts=1`
  because accepted points were double-counted as exhausted.

## [0.1.0] — 2026-04-10

Initial public release.

### Added
- Domain-agnostic N-dimensional parameter sweep engine (`run_sweep`) with
  parallel subprocess workers, eigenvalue continuation seeding, atomic
  checkpointing, adaptive timeout throttling.
- Evolutionary optimizer (`run_optimizer`) with coarse-then-fine grid scans,
  BLX-α crossover + mutation, multi-basin fine refinement, and Lamarckian
  feedback.
- `Problem` protocol for plugging in domain-specific evaluators.
- All grid quantization (`grid_resolution`) and fine-scan step/margin
  parameters are required inputs from the caller — the engine ships no
  domain defaults.
- Per-run `optimizer_state.json` checkpoint format with full provenance
  (settings, candidates, generation history, fitness trajectory).

[0.2.0]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.2.0
[0.1.1]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.1.1
[0.1.0]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.1.0
