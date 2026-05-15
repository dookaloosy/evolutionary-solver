# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-05-12

Breaking release. Candidate-level parallelism moves from threads to
processes — `Problem` subclasses must now be picklable. Consumers
pinned at `>=0.2,<0.3` must bump to `>=0.3,<0.4`.

### Changed (BREAKING)
- Candidate evaluation switched from `ThreadPoolExecutor` to
  `ProcessPoolExecutor`. Problem instances are pickled and sent to
  worker processes. Any subclass that holds unpicklable state
  (lambdas, open file handles, locks) must implement
  `__getstate__`/`__setstate__` or restructure.
- `Problem` picklability is now a documented contract of the engine.
  Base class provides `__getstate__`/`__setstate__`.
- Spinner/tick callbacks removed from worker processes — the parent
  prints one line per completed future instead.
- `n_workers` budget is now divided among `max_concurrent` candidates
  (`workers_per_cand = n_workers // max_concurrent`). Previously each
  candidate got the full worker pool.

### Added
- Phase 2 of `run_optimizer` tries `refine_basin()` per basin before
  building the fine grid. When the problem provides a refinement
  method, it bypasses the grid entirely — typically 100-200 function
  evaluations vs thousands of grid points.
- Fine-phase promotion (`n_fine_pts <= 1`) no longer bypasses
  `refine_basin()`. Trivial grids are always sent to the fine phase;
  `refine_basin()` runs first, and promotion only happens if the
  problem declines refinement.
- Enriched basin refinement output: start/finish prints with global
  elapsed time, coarse->refined fitness arrow, and problem-specific
  parameter summary via `format_best_point()`.
- `n_workers` and `max_concurrent` documented in `run_optimizer`
  docstring.

## [0.2.5] — 2026-05-12

### Added
- `Problem.refine_basin(center, bounds, output_dir)` — optional method
  for continuous basin refinement (e.g. Nelder-Mead). Returns
  `(fitness, best_point)` or `None` to fall back to the fine-grid sweep.
  Default implementation returns `None` (no new dependencies).
- `Problem.format_best_point(best_point)` — optional display method
  for refined parameter summaries. Default returns empty string.
- `Problem.describe_candidate(cand)` — optional one-line summary of
  a candidate's evolved values for display. Default returns empty string.
- 0-axis sweep support: problems with no searched axes (all parameters
  evolved by the GA) evaluate a single point per candidate.

### Fixed
- Coarse sweep and basin extraction now use the caller-provided
  `acceptance_threshold` instead of hardcoding 0. Previously, basins
  could form around zero-throughput coarse grid points, wasting
  compute on refinement of infeasible candidates.
- Penalty candidates (`coarse_fitness >= 1e6`) excluded from survivor
  selection. Previously the `max(2, ...`) floor could promote penalty
  candidates into the refinement phase, wasting compute on infeasible
  designs.
- Per-basin penalty skip: basins with `coarse_fitness >= 1e6` skip
  the fine-grid sweep entirely.
- Gaussian mutation for self-breed: when only one parent survives,
  continuous params get Gaussian noise (sigma = 10% of range width)
  instead of producing identical clones (BLX-alpha with d=0).

## [0.2.4] — 2026-05-03

### Fixed
- `make_grid`: single-point ranges (`start == stop`) now return
  `[start]` directly, bypassing snap-to-resolution that could round
  the value outside the valid range and produce an empty grid.
- `make_grid`: snap start/stop outward (floor/ceil to
  `grid_resolution`) before `arange`, then `np.unique` to deduplicate.
  Unaligned fine margins (e.g. +/-2.5 deg with 1.0 deg resolution) produced
  half-step values that snapped to duplicates, collapsing 6 grid
  points to 3.

## [0.2.3] — 2026-04-30

### Fixed
- Resume bug: candidates that completed fine scanning before a
  mid-generation interrupt were excluded from `coarse_ranked` (filtered
  on `status == 'coarse_done'` only). After resume, these candidates
  were missing from survivor selection and parent breeding, so their
  genes never propagated. The generation history also reported 1e6
  (penalty) as the best fitness despite a real winner existing. Now
  includes `fine_done` candidates in the coarse ranking.
- Atomic-write hardening: `save_state`, `save_summary`, and sweep
  results CSV now `fsync` before `os.replace` to prevent partial writes
  on crash/power-loss (WSL2 observed).

## [0.2.2] — 2026-04-26

### Fixed
- Coarse grid check: `len < 2` -> `len == 0`. Single-point axes (pinned
  parameters) now pass through instead of being culled as "too small".
- Fine scan: when the fine grid is 1x1 (all axes single-point or fully
  clamped), promote the coarse result directly instead of re-running an
  identical sweep. Finalize candidate status immediately when all basins
  are promoted.

## [0.2.1] — 2026-04-16

### Fixed
- Snap coarse-sweep grid to coarse-step multiples.

## [0.2.0] — 2026-04-15

Breaking release. Sub-sweep dimensionality is no longer hardcoded to 2.

### Changed (BREAKING)
- `run_optimizer` now accepts `fine_steps: dict[str, float]` and
  `fine_margins: dict[str, float]` keyed by axis name. The old
  positional kwargs `fine_step_0, fine_step_1, fine_margin_0,
  fine_margin_1` are **removed**. Sub-sweep dimensionality follows
  `len(problem.axis_names())`, so Problems can now expose 1, 2, 3, ...
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
  filtering; constant-factor speedup (2-3x in 2-D, shrinking with
  N). Docstring notes that proper high-N seeding needs a different
  algorithm (KD-tree / Poisson-disk) before extending to >=5 axes.

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
  candidates. Remaining `pop_size - N` candidates are sampled as before
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
  BLX-alpha crossover + mutation, multi-basin fine refinement, and Lamarckian
  feedback.
- `Problem` protocol for plugging in domain-specific evaluators.
- All grid quantization (`grid_resolution`) and fine-scan step/margin
  parameters are required inputs from the caller — the engine ships no
  domain defaults.
- Per-run `optimizer_state.json` checkpoint format with full provenance
  (settings, candidates, generation history, fitness trajectory).

[0.3.0]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.3.0
[0.2.5]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.2.5
[0.2.4]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.2.4
[0.2.3]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.2.3
[0.2.2]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.2.2
[0.2.1]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.2.1
[0.2.0]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.2.0
[0.1.1]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.1.1
[0.1.0]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.1.0
