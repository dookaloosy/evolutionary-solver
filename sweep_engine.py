"""Sweep engine for N-dimensional parameter sweeps.

Domain-agnostic: all domain knowledge lives in the Problem adapter.
Provides `run_sweep()` entry point and `Problem` base class.

The grid is defined by an OrderedDict of named axes:
    axes = OrderedDict([('x', x_arr), ('y', y_arr)])
Result arrays are N-dimensional: shape = tuple(len(v) for v in axes.values()).
Multi-indices are tuples: idx[k] indexes into axes_values[k].
"""
import sys, time, random, hashlib, csv, os, gc, collections
import json as json_mod
import multiprocessing
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product as iproduct

import numpy as np


# ── CLI utilities ────────────────────────────────────────────────────────────

def make_tick_callback(own_clock=False):
    """Create a tick callback for spinner animation.

    own_clock : bool — if True, use wall time from creation instead of
                the per-sweep elapsed value (avoids jumps when multiple
                sweeps share a single spinner).
    """
    _spinner = ['-', '\\', '|', '/']
    _idx = [0]
    _last_line_was_spinner = [False]
    _t0 = [time.perf_counter()] if own_clock else None

    def on_tick(elapsed):
        t = time.perf_counter() - _t0[0] if _t0 else elapsed
        c = _spinner[_idx[0] % 4]
        _idx[0] += 1
        if _last_line_was_spinner[0]:
            sys.stdout.write('\r')
        sys.stdout.write(f'[{t:6.0f}s] {c}')
        sys.stdout.flush()
        _last_line_was_spinner[0] = True

    on_tick.clear_spinner = lambda: (
        sys.stdout.write('\r\033[K') if _last_line_was_spinner[0] else None,
        _last_line_was_spinner.__setitem__(0, False),
    )

    on_tick.reset_clock = lambda: _t0.__setitem__(0, time.perf_counter()) if _t0 else None

    return on_tick


# ── Problem protocol ────────────────────────────────────────────────────────

class Problem:
    """Interface for domain-specific sweep/optimizer behaviour.

    Subclass this to adapt the sweep engine to a new domain.  The sweep
    engine calls these methods instead of hard-coding domain physics.

    Lifecycle: construct → prepare(params) → evaluate() per point.
    """

    # ── Identity ──────────────────────────────────────────────────────────

    name = ''
    """Short string identifying the problem variant. Used in auto-generated
    run names (e.g. 'optim_<name>_<timestamp>')."""

    # ── Result protocol ──────────────────────────────────────────────────
    # Subclass should override these to define result tuple structure.

    result_names = ('val0', 'val1', 'val2', 'val3', 'val4', 'val5')
    """Names of fields in the result tuple returned by evaluate()."""

    acceptance_field = 'val4'
    """Which result field is compared against acceptance_threshold."""

    # ── Lifecycle ────────────────────────────────────────────────────────

    def prepare(self, params):
        """One-time setup before sweep. Store params, build worker config, etc.

        Called once before the sweep loop starts. After this, evaluate(),
        validate_point(), valid_range(), seed(), axis_names(), etc. can be
        called without passing params again.
        """
        pass

    def evaluate(self, axis_values, seed, timeout):
        """Run one grid point.

        Parameters
        ----------
        axis_values : tuple of float — physical axis values
        seed : tuple (lam_re, lam_im) — eigenvalue/continuation seed
        timeout : float — max seconds

        Returns
        -------
        (result_tuple, 'ok') on success, where result_tuple matches
        result_names in length.
        (None, reason_string) on failure.
        """
        raise NotImplementedError

    # ── Grid validation ──────────────────────────────────────────────────

    def validate_point(self, ax_vals):
        """Check if a grid point is valid.

        Returns 0 if valid, or a negative n_attempts code:
        -1 = invalid configuration, -5 = below min feature size.
        """
        return 0

    def valid_range(self):
        """Return per-axis {name_min, name_max} dict, or None if infeasible.

        Used by the optimizer to determine the sweep region for a candidate.
        """
        return None

    # ── Fitness extraction ───────────────────────────────────────────────

    def fitness(self, stats, acceptance_threshold=0.10):
        """Extract (fitness_value, best_point_dict) or (None, None)."""
        raise NotImplementedError

    def fitness_basins(self, stats, acceptance_threshold=0.10, n_basins=1,
                       min_separation=2):
        """Extract top-K basins as list of (fitness, best_point) tuples."""
        raise NotImplementedError

    # ── Seeding ──────────────────────────────────────────────────────────

    def seed(self):
        """Return default eigenvalue/continuation seed (lam_re, lam_im)."""
        return (0.0, 0.0)

    def compute_seed_from_neighbors(self, neighbor_results, weights,
                                    default_seed):
        """Compute seed from accepted neighbor results.

        Parameters
        ----------
        neighbor_results : list of float — primary result values from neighbors
        weights : list of float — importance weights (e.g. 1/|secondary|)
        default_seed : tuple — fallback if no neighbors

        Returns (lam_re, lam_im) seed tuple.
        """
        return default_seed

    # ── Axis metadata ────────────────────────────────────────────────────

    def axis_names(self):
        """Return ordered list of searched axis names, e.g. ['x', 'y']."""
        raise NotImplementedError

    def axis_metadata(self):
        """Return list of axis dicts: [{'name': ..., 'unit': ...}, ...]."""
        raise NotImplementedError

    # ── Config / hashing ─────────────────────────────────────────────────

    def physics_params(self):
        """Return dict of hashable physics parameters for config hash."""
        raise NotImplementedError

    def config_hash(self):
        """Compute config hash from physics params."""
        meta = self.physics_params()
        return hashlib.sha256(
            json_mod.dumps(meta, sort_keys=True).encode()
        ).hexdigest()[:8]

    def save_config(self, output_dir, config_hash, searched_axes,
                    acceptance_threshold, max_attempts, grid_resolution):
        """Write config.json to output_dir."""
        raise NotImplementedError

    def load_config(self, output_dir):
        """Load config.json. Returns (problem_name, params, grid, sweep_opts)."""
        raise NotImplementedError

    # ── Display formatting ───────────────────────────────────────────────

    def format_fitness(self, value):
        """Format a fitness value for display."""
        return f'{value:.4f}'

    def format_point(self, name, value):
        """Format a single axis value for display."""
        return f'{name}={value}'

    def format_bounds(self, name, lo, hi, unit=''):
        """Format axis bounds for display."""
        return f'{name}=[{lo}, {hi}]'

    # ── Optimizer helpers ────────────────────────────────────────────────

    def round_param(self, name, value):
        """Round an evolved parameter to appropriate precision."""
        return value

    def coarsen_params(self, factor):
        """Return params dict modified for coarse evaluation.

        Called before a coarse sweep. The default returns params unchanged.
        Override to e.g. increase mesh element size by *factor*.
        """
        return dict(self._params) if hasattr(self, '_params') else {}

    # ── Stats / export ───────────────────────────────────────────────────

    def extra_stats(self, searched_axes):
        """Return dict of domain-specific entries to add to stats."""
        return {}

    def export_results(self, output_dir, searched_axes, stats):
        """Write domain-specific CSV/PNG exports."""
        pass


# ── N-dim grid helpers ──────────────────────────────────────────────────────

def _adjacent(idx, shape):
    """Yield 2N-connected neighbor indices of *idx* in an N-dim grid."""
    for d in range(len(idx)):
        for delta in (-1, 1):
            new = list(idx)
            new[d] += delta
            if 0 <= new[d] < shape[d]:
                yield tuple(new)


def _chebyshev_shell(center, r, shape):
    """Yield all in-bounds indices at Chebyshev distance exactly *r* from *center*.

    Enumerates the shell surface directly rather than iterating the full
    (2r+1)^ndim cube and filtering — for each axis k and sign ±, fix
    offsets[k] = ±r, let the other axes range over [-r, r], and dedupe
    corners/edges via a set. Shell size ≈ 2·ndim·(2r+1)^(ndim-1); total
    work across r = 1..R telescopes to (2R+1)^ndim rather than the sum
    of cubes it used to be.

    Savings are largest in low dimensions (2-3 axes: ~2× speedup over
    the cube-and-filter loop) and shrink as N grows (at N ≥ 5 the shell
    IS most of the cube).

    TODO(high-N seeding): the shell surface itself is ~(2r+1)^(ndim-1)
    — exponential in ndim. Beyond ~4 search axes, grid-shell enumeration
    for continuation seeding (find_nearest_seed) becomes intractable
    regardless of how efficient this generator is. Switch to a proper
    nearest-neighbour data structure (KD-tree over accepted points,
    Poisson-disk sampling, or a sparse index) before extending this
    engine to 5+ axes. This rewrite only buys headroom; it does not
    fix the asymptotic scaling.
    """
    ndim = len(center)
    if ndim == 0 or r == 0:
        if all(0 <= center[d] < shape[d] for d in range(ndim)):
            yield tuple(center)
        return
    seen = set()
    for k in range(ndim):
        for sign in (-1, 1):
            fixed = center[k] + sign * r
            if not (0 <= fixed < shape[k]):
                continue
            # Other axes sweep the full [-r, r] range; dedup via `seen`
            # handles corners where two+ axes both sit at ±r.
            ranges = []
            for j in range(ndim):
                if j == k:
                    ranges.append((fixed,))
                else:
                    lo = max(0, center[j] - r)
                    hi = min(shape[j] - 1, center[j] + r)
                    ranges.append(tuple(range(lo, hi + 1)))
            for combo in iproduct(*ranges):
                if combo in seen:
                    continue
                seen.add(combo)
                yield combo


def _all_indices(shape):
    """Yield all multi-indices for an N-dim grid."""
    return iproduct(*[range(s) for s in shape])


def extract_basins(fitness_map, k=1, min_separation=2):
    """Find up to *k* local minima in an N-dim fitness array.

    Parameters
    ----------
    fitness_map : np.ndarray — fitness values (lower = better).
        NaN and inf entries are ignored.
    k : int — maximum number of basins to return.
    min_separation : int — minimum Chebyshev distance (in grid cells)
        between selected basins.  Prevents redundant zooms on the
        same valley.

    Returns
    -------
    list of tuples — multi-indices of the selected minima, ordered by
        fitness (best first).  Empty list if no finite values exist.
    """
    shape = fitness_map.shape
    finite = np.isfinite(fitness_map)
    if not np.any(finite):
        return []

    # Find all local minima: points where no Chebyshev-1 neighbor is better.
    local_min = []
    for idx in _all_indices(shape):
        if not finite[idx]:
            continue
        val = fitness_map[idx]
        is_min = True
        for nb in _adjacent(idx, shape):
            if finite[nb] and fitness_map[nb] < val:
                is_min = False
                break
        if is_min:
            local_min.append((val, idx))

    # If no local minima found (e.g. flat region), fall back to global argmin.
    if not local_min:
        idx = np.unravel_index(np.nanargmin(fitness_map), shape)
        return [idx]

    # Sort by fitness (best first).
    local_min.sort(key=lambda x: x[0])

    # Greedy selection: pick best, skip any within min_separation.
    selected = []
    for val, idx in local_min:
        if len(selected) >= k:
            break
        too_close = False
        for _, prev in selected:
            if max(abs(a - b) for a, b in zip(idx, prev)) < min_separation:
                too_close = True
                break
        if not too_close:
            selected.append((val, idx))

    return [idx for _, idx in selected]


def make_grid(start, stop, step, grid_resolution):
    """Create a grid array snapped to *grid_resolution*.

    Snapping prevents float drift between independent runs from
    aliasing the same logical grid point to slightly different values
    on resume.
    """
    arr = np.arange(start, stop + step * 0.5, step)
    return np.round(arr / grid_resolution) * grid_resolution


def _row_to_idx(row, axis_names_list, axis_arrays, grid_resolution):
    """Map a CSV row to a multi-index in the grid. Returns None if out of range.

    Natural order: idx[k] indexes into axis k.  Match tolerance scales
    with *grid_resolution* (10% of one snap step).
    """
    tol = grid_resolution * 0.1
    ndim = len(axis_names_list)
    idx = [0] * ndim
    for k, name in enumerate(axis_names_list):
        val = float(row.get(name, 0))
        match = np.argmin(np.abs(axis_arrays[k] - val))
        if abs(axis_arrays[k][match] - val) >= tol:
            return None
        idx[k] = match
    return tuple(idx)



# ── Worker dispatch (module-level for pickling) ──────────────────────────────

def _run_one_point(args):
    """Evaluate a single grid point via problem.evaluate(). Module-level for pickling.

    args = (idx, axis_values, lam_re, lam_im, problem, timeout)
    """
    idx, axis_values, lam_re, lam_im, problem, timeout = args
    result, reason = problem.evaluate(axis_values, (lam_re, lam_im), timeout)
    return (idx, result, reason)


# ── Sweep engine ─────────────────────────────────────────────────────────────

def run_sweep(
    searched_axes, output_dir, problem,
    grid_resolution,
    on_progress=None, on_tick=None, n_workers=None,
    acceptance_threshold=0.10, max_attempts=4,
    progress_interval=10, worker_timeout=120, max_timeouts=3,
    export=True,
):
    """Run an N-dimensional parameter sweep.

    Parameters
    ----------
    searched_axes : OrderedDict[str, np.ndarray] — named sweep axes
    output_dir : str — output directory path
    problem : Problem — domain adapter (must have prepare() called already)
    grid_resolution : float — minimum quantization step for grid axes,
        used for CSV-row matching tolerance on resume.
    acceptance_threshold : float — quality metric threshold for acceptance
    max_attempts : int — max retries per point

    Returns dict with final stats.
    """
    if grid_resolution is None:
        raise ValueError('run_sweep requires grid_resolution')
    # ── Result field setup ───────────────────────────────────────────────
    rnames = problem.result_names
    n_fields = len(rnames)
    accept_idx = rnames.index(problem.acceptance_field)

    # ── Grid shape ───────────────────────────────────────────────────────
    # Natural order: shape matches OrderedDict axis order.
    # idx[k] indexes into axis k.  Visualization code transposes for plotting.
    axis_names_list = list(searched_axes.keys())
    axis_arrays = list(searched_axes.values())
    ndim = len(searched_axes)
    shape = tuple(len(a) for a in axis_arrays)
    n_total = 1
    for s in shape:
        n_total *= s

    def _idx_to_axis_vals(idx):
        return tuple(axis_arrays[d][idx[d]] for d in range(ndim))

    # ── Config ───────────────────────────────────────────────────────────
    config_hash = problem.config_hash()
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, 'results.csv')

    problem.save_config(output_dir, config_hash, searched_axes,
                        acceptance_threshold, max_attempts, grid_resolution)

    default_seed = problem.seed()

    # ── Sweep constants ──────────────────────────────────────────────────
    MAX_TIMEOUTS = max_timeouts
    MAX_ATTEMPTS = max_attempts
    TIMEOUT_COOLDOWN = 30.0

    # ── Per-point storage (one map per result field) ─────────────────────
    result_maps = {name: np.full(shape, np.nan) for name in rnames}
    n_attempts_map = np.zeros(shape, dtype=np.int32)
    timeout_map = np.zeros(shape, dtype=np.int32)
    elapsed_map = np.full(shape, np.nan)
    cooldown_until = {}

    # Convenience references for acceptance checking
    accept_map = result_maps[problem.acceptance_field]
    primary_map = result_maps[rnames[0]]

    # ── Load existing results ────────────────────────────────────────────
    n_loaded = 0
    _out_of_range_rows = []
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = _row_to_idx(row, axis_names_list, axis_arrays,
                                  grid_resolution)
                if idx is not None:
                    if row.get(rnames[0], ''):
                        for fi, rn in enumerate(rnames):
                            val = row.get(rn, '')
                            if val:
                                result_maps[rn][idx] = float(val)
                    n_attempts_map[idx] = int(row['n_attempts'])
                    n_loaded += 1
                else:
                    _out_of_range_rows.append(row)
        print(f'Loaded {n_loaded} points from {results_file}')
        if _out_of_range_rows:
            print(f'  ({len(_out_of_range_rows)} out-of-range points preserved)')
    else:
        print(f'Starting fresh (no {results_file})')

    # ── Pre-filter: configuration validity ─────────────────────────────
    # Clear previous min-feature-size flags (-5) so they get re-evaluated
    # (min_feature_size may have changed between runs)
    n_cleared = int(np.sum(n_attempts_map == -5))
    if n_cleared > 0:
        n_attempts_map[n_attempts_map == -5] = 0
        print(f'Cleared {n_cleared} previous min-feature-size flags (re-validating)')
    n_prefiltered = 0
    n_below_min_feature = 0
    for idx in _all_indices(shape):
        if n_attempts_map[idx] != 0:
            continue
        ax_vals = _idx_to_axis_vals(idx)
        code = problem.validate_point(ax_vals)
        if code == -5:
            n_attempts_map[idx] = -5
            n_below_min_feature += 1
        elif code < 0:
            n_attempts_map[idx] = code
            n_prefiltered += 1
    print(f'Pre-filtered {n_prefiltered} invalid points')
    if n_below_min_feature > 0:
        print(f'Marked {n_below_min_feature} points below min feature size')

    # ── Save results (atomic) ────────────────────────────────────────────
    def save_results():
        import tempfile, shutil
        fd, tmp_path = tempfile.mkstemp(
            suffix='.csv', dir=os.path.dirname(results_file) or '.')
        try:
            with os.fdopen(fd, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(axis_names_list) + list(rnames) +
                                ['n_attempts'])
                for idx in _all_indices(shape):
                    ax_vals = _idx_to_axis_vals(idx)
                    if not np.isnan(primary_map[idx]):
                        writer.writerow(
                            [f'{v:.7e}' for v in ax_vals] +
                            [f'{result_maps[rn][idx]:.10e}' if fi < 4
                             else f'{result_maps[rn][idx]:.6f}'
                             for fi, rn in enumerate(rnames)] +
                            [n_attempts_map[idx]])
                    elif (n_attempts_map[idx] < 0 or
                          n_attempts_map[idx] >= MAX_ATTEMPTS):
                        writer.writerow(
                            [f'{v:.7e}' for v in ax_vals] +
                            ['' for _ in rnames] +
                            [n_attempts_map[idx]])
                for oor in _out_of_range_rows:
                    row_vals = [oor.get(name, '') for name in axis_names_list]
                    row_vals += [oor.get(rn, '') for rn in rnames]
                    row_vals += [oor.get('n_attempts', '')]
                    writer.writerow(row_vals)
            shutil.move(tmp_path, results_file)
        except BaseException:
            os.unlink(tmp_path)
            raise

    # ── Candidate selection (N-dim) ──────────────────────────────────────
    def _is_candidate(idx, now):
        return ((np.isnan(primary_map[idx]) or
                 accept_map[idx] < acceptance_threshold)
                and 0 <= n_attempts_map[idx] < MAX_ATTEMPTS
                and timeout_map[idx] < MAX_TIMEOUTS
                and cooldown_until.get(idx, 0) <= now
                and idx not in in_flight)

    def pick_random_candidate():
        now = time.monotonic()
        nearby = []
        faroff = []
        for idx in _all_indices(shape):
            if _is_candidate(idx, now):
                if find_nearest_seed(idx) is not None:
                    nearby.append(idx)
                else:
                    faroff.append(idx)
        if nearby:
            return random.choice(nearby)
        if faroff:
            return random.choice(faroff)
        return None

    def pick_adjacent_candidate(idx0):
        now = time.monotonic()
        candidates = [nb for nb in _adjacent(idx0, shape)
                      if _is_candidate(nb, now)]
        return random.choice(candidates) if candidates else None

    # ── Continuation seeding (N-dim Chebyshev shell) ─────────────────────
    def find_nearest_seed(idx0, max_r=5):
        n_extra = n_attempts_map[idx0]
        first_ring = None
        primary_vals = []
        weights = []
        for r in range(1, max_r + 1):
            ring_vals = []
            ring_w = []
            for nb in _chebyshev_shell(idx0, r, shape):
                if (not np.isnan(primary_map[nb])
                        and accept_map[nb] >= acceptance_threshold):
                    ring_vals.append(primary_map[nb])
                    ring_w.append(
                        1.0 / max(abs(result_maps[rnames[1]][nb]), 1e-12))
            if ring_vals:
                primary_vals.extend(ring_vals)
                weights.extend(ring_w)
                if first_ring is None:
                    first_ring = r
                if r - first_ring >= n_extra:
                    break
        if primary_vals:
            return problem.compute_seed_from_neighbors(
                primary_vals, weights, default_seed)
        return None

    def seed_for(idx):
        s = find_nearest_seed(idx)
        return s if s is not None else default_seed

    # ── Timeout throttle ─────────────────────────────────────────────────
    def compute_effective_max():
        if not recent_results:
            return N_WORKERS
        n_recent_timeouts = sum(1 for r in recent_results if r == 'timeout')
        timeout_frac = n_recent_timeouts / len(recent_results)
        if timeout_frac > 0.5:
            return max(2, N_WORKERS // 4)
        elif timeout_frac > 0.25:
            return max(2, N_WORKERS // 2)
        return N_WORKERS

    # ── Main loop ────────────────────────────────────────────────────────
    N_WORKERS = n_workers or min(
        max(1, int(multiprocessing.cpu_count() * 0.5)), n_total)

    converged_init = ~np.isnan(primary_map)
    accepted_init = converged_init & (accept_map >= acceptance_threshold)
    n_accepted_init = int(np.sum(accepted_init))
    n_skipped = int(np.sum(n_attempts_map < 0))
    n_exhausted = int(np.sum(n_attempts_map >= MAX_ATTEMPTS))
    n_pending = int(np.sum(
        (~accepted_init) & (n_attempts_map >= 0) &
        (n_attempts_map < MAX_ATTEMPTS)))
    shape_str = ' x '.join(str(s) for s in shape)
    print(f'Grid: {shape_str} = {n_total} points, {N_WORKERS} workers')
    print(f'Already accepted: {n_accepted_init}, pending: {n_pending}, '
          f'skipped: {n_skipped}, exhausted: {n_exhausted}')

    if n_pending == 0:
        print('No pending points — nothing to do')
        return _make_stats(result_maps, n_attempts_map, timeout_map,
                           0, 0, {}, 0, N_WORKERS, 0.0,
                           searched_axes, acceptance_threshold,
                           max_attempts, problem)

    t_start = time.perf_counter()
    n_done = 0; n_fail = 0
    fail_counts = {}
    in_flight = set()
    recent_results = collections.deque(maxlen=N_WORKERS * 2)
    effective_max = N_WORKERS

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {}
        submit_times = {}

        def submit_point(idx, max_concurrent=None):
            cap = max_concurrent if max_concurrent is not None else N_WORKERS
            if len(in_flight) >= cap:
                return False
            in_flight.add(idx)
            s = seed_for(idx)
            ax_vals = _idx_to_axis_vals(idx)
            job = (idx, ax_vals, s[0], s[1], problem, worker_timeout)
            fut = executor.submit(_run_one_point, job)
            futures[fut] = idx
            submit_times[fut] = time.monotonic()
            return True

        def submit_random():
            pt = pick_random_candidate()
            if pt is None:
                return False
            submit_point(pt, max_concurrent=effective_max)
            return True

        # Fill initial batch
        for _ in range(min(N_WORKERS, n_pending)):
            if not submit_random():
                break

        stall_count = 0
        while True:
            if not futures and not in_flight:
                any_left = any(
                    _is_candidate(idx, time.monotonic())
                    for idx in _all_indices(shape)
                )
                if any_left:
                    time.sleep(TIMEOUT_COOLDOWN + 1)
                    if submit_random():
                        stall_count = 0
                        continue
                    stall_count += 1
                    if stall_count >= 3:
                        break
                    continue
                break

            done_futs = []
            try:
                for fut in as_completed(futures, timeout=TIMEOUT_COOLDOWN):
                    done_futs.append(fut)
                    break
            except TimeoutError:
                pass

            if not done_futs:
                if on_tick:
                    on_tick(time.perf_counter() - t_start)
                while len(in_flight) < effective_max:
                    if not submit_random():
                        break
                continue

            for fut in done_futs:
                idx = futures.pop(fut)
                in_flight.discard(idx)
                elapsed_pt = time.monotonic() - submit_times.pop(
                    fut, time.monotonic())
                _, result, reason = fut.result()
                elapsed_map[idx] = elapsed_pt

                # Timeout / attempt accounting
                if reason == 'timeout':
                    timeout_map[idx] += 1
                    if timeout_map[idx] >= MAX_TIMEOUTS:
                        n_attempts_map[idx] = -3
                    cooldown_until[idx] = (
                        time.monotonic() + TIMEOUT_COOLDOWN * timeout_map[idx])
                    for nb in _adjacent(idx, shape):
                        cooldown_until[nb] = max(
                            cooldown_until.get(nb, 0),
                            time.monotonic() + TIMEOUT_COOLDOWN)
                elif reason.startswith('exception'):
                    pass
                else:
                    n_attempts_map[idx] += 1
                    if n_attempts_map[idx] >= MAX_ATTEMPTS and result is None:
                        n_attempts_map[idx] = -2

                recent_results.append(reason)
                effective_max = compute_effective_max()

                if result is not None:
                    for fi, rn in enumerate(rnames):
                        result_maps[rn][idx] = result[fi]
                    n_done += 1

                    adj = pick_adjacent_candidate(idx)
                    if adj is not None:
                        submit_point(adj, max_concurrent=effective_max)
                    else:
                        submit_random()
                else:
                    n_fail += 1
                    fail_counts[reason] = fail_counts.get(reason, 0) + 1
                    if reason == 'invalid':
                        n_attempts_map[idx] = -1
                    elif reason.startswith('exception'):
                        n_attempts_map[idx] = -4
                    if reason == 'timeout':
                        submit_random()
                    else:
                        adj = pick_adjacent_candidate(idx)
                        if adj is not None:
                            submit_point(adj, max_concurrent=effective_max)
                        else:
                            submit_random()

                if on_tick:
                    on_tick(time.perf_counter() - t_start)

                # Checkpoint + progress
                total = n_done + n_fail
                if total % 10 == 0 and total > 0:
                    save_results()
                if total % progress_interval == 0 and total > 0:
                    if on_progress:
                        on_progress(_make_stats(
                            result_maps, n_attempts_map, timeout_map,
                            n_done, n_fail, fail_counts,
                            len(in_flight), effective_max,
                            time.perf_counter() - t_start,
                            searched_axes, acceptance_threshold,
                            max_attempts, problem))

    dt = time.perf_counter() - t_start
    save_results()

    # ── Domain-specific exports (CSV/PNG) ────────────────────────────────
    stats = _make_stats(result_maps, n_attempts_map, timeout_map,
                        n_done, n_fail, fail_counts,
                        0, effective_max, dt,
                        searched_axes, acceptance_threshold,
                        max_attempts, problem)
    if export:
        problem.export_results(output_dir, searched_axes, stats)

    n_accepted_final = int(np.sum(
        (~np.isnan(primary_map)) & (accept_map >= acceptance_threshold)))
    n_converged_final = int(np.sum(~np.isnan(primary_map)))
    if on_tick and hasattr(on_tick, 'clear_spinner'):
        on_tick.clear_spinner()
    print(f'Sweep done in {dt:.1f}s: {n_converged_final} converged, '
          f'{n_accepted_final} accepted (threshold>={acceptance_threshold}), '
          f'{n_fail} failures, {int(timeout_map.sum())} timeouts')

    return stats


def _make_stats(result_maps, n_attempts_map, timeout_map,
                n_done, n_fail, fail_counts,
                in_flight, effective_max, elapsed,
                searched_axes, acceptance_threshold, max_attempts,
                problem=None):
    """Build stats dict from sweep state."""
    stats = dict(result_maps)  # {field_name: ndarray, ...}
    # Also store with _map suffix for consumers that expect it
    for name, arr in result_maps.items():
        stats[f'{name}_map'] = arr
    stats.update({
        'n_attempts_map': n_attempts_map,
        'timeout_map': timeout_map,
        'n_done': n_done,
        'n_fail': n_fail,
        'fail_counts': fail_counts,
        'in_flight': in_flight,
        'effective_max': effective_max,
        'elapsed': elapsed,
        'searched_axes': searched_axes,
        'acceptance_threshold': acceptance_threshold,
        'max_attempts': max_attempts,
    })
    # Domain-specific extras from problem
    if problem is not None:
        stats.update(problem.extra_stats(searched_axes))
    return stats
