# CLAUDE.md

## Repository Purpose

Domain-agnostic grid sweep + evolutionary optimizer with continuation seeding. Designed for expensive objective functions with spatial structure (eigenvalue/PDE problems).

## Architecture

Two modules, zero domain imports:

- **`sweep_engine.py`** — `Problem` base class (adapter protocol) + `run_sweep()` entry point. N-dimensional grid sweep with Chebyshev-shell neighbor seeding. Natural array order: `shape = (len(ax0), len(ax1))`, `idx[k]` indexes axis `k`. Checkpoint I/O via results.csv. `extract_basins()` for multi-basin minimum detection. `make_tick_callback()` for terminal spinner.
- **`optimizer_engine.py`** — `run_optimizer()` evolutionary search over outer parameters. Coarse-then-fine multi-resolution sweeps. BLX-alpha crossover + mutation. Multi-basin fine refinement. Population management (LHS init, elitism, immigrants). Checkpointing via JSON state files.

## Problem Protocol

Subclass `Problem` from `sweep_engine` and implement:

**Required** (raise NotImplementedError):
- `evaluate(axis_values, seed, timeout)` — run one grid point, return `(result_tuple, 'ok')` or `(None, reason)`
- `axis_names()` — ordered list of searched axis names
- `axis_metadata()` — list of `{name, unit}` dicts
- `physics_params()` — dict for config hashing
- `save_config(output_dir, config_hash, searched_axes, acceptance_threshold, max_attempts, grid_resolution)`
- `load_config(output_dir)` — returns `(geometry, params, grid, sweep_opts)`
- `fitness(stats, acceptance_threshold)` — extract `(value, best_point_dict)` or `(None, None)`
- `fitness_basins(stats, acceptance_threshold, n_basins, min_separation)` — list of `(value, best_point)` tuples

**Required attributes**:
- `result_names` — tuple of field name strings for the result tuple
- `acceptance_field` — which result field is compared against `acceptance_threshold`

**Optional** (sensible defaults):
- `prepare(params)` — one-time setup before sweep
- `validate_point(ax_vals)` — return 0 (valid), -1 (invalid geometry), -5 (below min feature size)
- `valid_range()` — `{name_min, name_max, ...}` dict for optimizer bounds
- `seed()` — default continuation seed `(lam_re, lam_im)`
- `compute_seed_from_neighbors(vals, weights, default)` — seed from neighbor results
- `coarsen_params(factor)` — return params for coarse evaluation
- `round_param(name, value)` — rounding precision for evolved params
- `format_fitness(value)`, `format_point(name, value)`, `format_bounds(name, lo, hi, unit)` — display
- `extra_stats(searched_axes)` — domain-specific entries for stats dict
- `export_results(output_dir, searched_axes, stats)` — CSV/PNG exports

## Install

```bash
pip install -e .
```

