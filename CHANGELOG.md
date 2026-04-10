# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/dookaloosy/evolutionary-solver/releases/tag/v0.1.0
