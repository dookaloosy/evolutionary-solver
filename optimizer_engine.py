"""Evolutionary optimizer — domain-agnostic.

Searches evolved parameter space using a coarse-then-fine multi-resolution
sweep strategy. Each candidate is evaluated by running sweep_engine.run_sweep()
over the valid search region, with fitness extracted by the Problem adapter.

Usage:
    from optimizer_engine import run_optimizer
    run_optimizer(fixed_params, evolved_params, problem=problem, ...)
"""
import os
import json as json_mod
import time
import random
from collections import OrderedDict

import numpy as np

from sweep_engine import run_sweep, make_grid, extract_basins, make_tick_callback


# ── Population management ────────────────────────────────────────────────────

def init_population(evolved_params, pop_size, fixed_params, problem=None):
    """Initialize population via Latin hypercube sampling.

    evolved_params: dict of param_name → (min, max) for continuous,
                  or param_name → [list of choices] for discrete.
    """
    candidates = []
    continuous = {k: v for k, v in evolved_params.items() if isinstance(v, tuple)}
    discrete = {k: v for k, v in evolved_params.items() if isinstance(v, list)}

    n_cont = len(continuous)
    # Latin hypercube for continuous params
    if n_cont > 0:
        intervals = np.linspace(0, 1, pop_size + 1)
        lhs = np.zeros((pop_size, n_cont))
        for j in range(n_cont):
            perm = np.random.permutation(pop_size)
            for i in range(pop_size):
                lhs[perm[i], j] = np.random.uniform(intervals[i], intervals[i+1])

    for i in range(pop_size):
        outer = {}
        for j, (k, (lo, hi)) in enumerate(continuous.items()):
            val = lo + lhs[i, j] * (hi - lo)
            outer[k] = problem.round_param(k, val) if problem else val
        for k, choices in discrete.items():
            outer[k] = random.choice(choices)

        full_params = dict(fixed_params)
        full_params.update(outer)
        candidates.append(_make_candidate(f'gen00_cand{i:02d}', outer, full_params))

    return candidates


def breed(parents, evolved_params, fixed_params, gen, n_children, alpha=0.5,
          start_idx=0, problem=None):
    """Breed children from parent candidates via BLX-α crossover + mutation."""
    continuous = {k: v for k, v in evolved_params.items() if isinstance(v, tuple)}
    discrete = {k: v for k, v in evolved_params.items() if isinstance(v, list)}
    children = []

    for ci in range(n_children):
        if len(parents) >= 2:
            p1, p2 = random.sample(parents, 2)
        else:
            p1 = p2 = parents[0]  # self-breed with mutation
        outer = {}

        # BLX-α crossover for continuous
        for k, (lo, hi) in continuous.items():
            v1 = p1['evolved_values'][k]
            v2 = p2['evolved_values'][k]
            d = abs(v1 - v2)
            child_lo = min(v1, v2) - alpha * d
            child_hi = max(v1, v2) + alpha * d
            child_lo = max(child_lo, lo)
            child_hi = min(child_hi, hi)
            val = np.random.uniform(child_lo, child_hi)
            outer[k] = problem.round_param(k, val) if problem else val

        # Discrete: random pick from parents
        for k, choices in discrete.items():
            outer[k] = random.choice([p1['evolved_values'][k], p2['evolved_values'][k]])
            # Small mutation probability
            if random.random() < 0.1:
                idx = choices.index(outer[k])
                delta = random.choice([-1, 1])
                new_idx = max(0, min(len(choices) - 1, idx + delta))
                outer[k] = choices[new_idx]

        full_params = dict(fixed_params)
        full_params.update(outer)
        cid = f'gen{gen:02d}_cand{start_idx + ci:02d}'
        child = _make_candidate(cid, outer, full_params,
                                parents=[p1['id'], p2['id']])
        children.append(child)

    return children


def _make_candidate(cid, evolved_values, full_params, parents=None):
    return {
        'id': cid,
        'evolved_values': dict(evolved_values),
        'full_params': dict(full_params),
        'config_hash': None,
        'output_dir': None,
        'coarse_fitness': None,
        'fine_fitness': None,
        'valid_range': None,
        'best_point': None,
        'parents': parents or [],
        'status': 'pending',
    }


# ── Checkpoint I/O ───────────────────────────────────────────────────────────

def save_state(state, run_dir):
    """Save optimizer state to JSON (atomic write)."""
    path = os.path.join(run_dir, 'optimizer_state.json')
    tmp = path + '.tmp'

    def _default(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return {'__complex__': True, 're': obj.real, 'im': obj.imag}
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(tmp, 'w') as f:
        json_mod.dump(state, f, indent=2, default=_default)
    os.replace(tmp, path)


def load_state(run_dir):
    """Load optimizer state from JSON. Returns None if no state file."""
    path = os.path.join(run_dir, 'optimizer_state.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json_mod.load(f)


def save_summary(state, run_dir):
    """Write summary.csv: one row per generation with best fitness."""
    path = os.path.join(run_dir, 'summary.csv')
    with open(path, 'w') as f:
        f.write('generation,best_coarse,best_fine,best_candidate,best_params\n')
        for gen_info in state.get('generation_history', []):
            f.write(f"{gen_info['generation']},"
                    f"{gen_info.get('best_coarse', '')},"
                    f"{gen_info.get('best_fine', '')},"
                    f"{gen_info.get('best_id', '')},"
                    f"\"{gen_info.get('best_outer', '')}\"\n")


def find_winner(state):
    """Find the overall winner (lowest fine_fitness among fine_done candidates).

    Returns the candidate dict, or None if no candidates qualify.
    """
    done = [c for c in state['candidates']
            if c.get('fine_fitness') is not None and c['status'] == 'fine_done']
    if not done:
        return None
    return min(done, key=lambda c: c['fine_fitness'])


# ── Main optimizer loop ──────────────────────────────────────────────────────

def run_optimizer(
    fixed_params,
    evolved_params,
    grid_resolution,
    fine_step_0,
    fine_step_1,
    fine_margin_0,
    fine_margin_1,
    run_name=None,
    pop_size=8,
    max_generations=20,
    coarse_factor=2,
    acceptance_threshold=0.10,
    max_attempts=3,
    n_workers=None,
    max_concurrent=2,
    convergence_patience=3,
    convergence_threshold=0.01,
    on_generation=None,
    n_basins=1,
    problem=None,
    seed_candidates=None,
):
    """Run evolutionary optimization.

    Parameters
    ----------
    fixed_params : dict — domain-specific fixed parameters
    evolved_params : dict — param_name → (min, max) or [choices]
    grid_resolution : float — minimum quantization step for grid axes
        (e.g. 1e-6 for um-scale, 1e-9 for nm-scale problems). Used to
        snap grid points and avoid float aliasing on resume.
    fine_step_0, fine_step_1 : float — fine-scan step size for axes 0/1
    fine_margin_0, fine_margin_1 : float — fine-scan zoom half-width for axes 0/1
    problem : Problem — domain adapter (must have .name set)
    """
    if any(v is None for v in (grid_resolution, fine_step_0, fine_step_1,
                                fine_margin_0, fine_margin_1)):
        raise ValueError(
            'run_optimizer requires grid_resolution and all four of '
            'fine_step_0, fine_step_1, fine_margin_0, fine_margin_1')
    # Get axis names from problem
    problem.prepare(fixed_params)
    axis_names_list = problem.axis_names()
    axis_meta = problem.axis_metadata()
    a0, a1 = axis_names_list[0], axis_names_list[1]

    import threading
    _state_lock = threading.Lock()

    if run_name is None:
        run_name = f'optim_{problem.name}_{time.strftime("%Y%m%d_%H%M%S")}'
    run_dir = os.path.join('output', run_name)
    os.makedirs(run_dir, exist_ok=True)

    # ── Resume from checkpoint ───────────────────────────────────────────
    state = load_state(run_dir)
    if state is not None:
        print(f"Resuming from {run_dir}: generation {state['current_gen']}")
        candidates_all = state['candidates']
        gen_start = state['current_gen']
        best_fitness_history = state.get('best_fitness_history', [])
        # Reproducibility: saved grid/step values win over runtime args
        saved = state['settings']
        grid_resolution = saved['grid_resolution']
        fine_step_0 = saved[f'fine_step_{a0}']
        fine_step_1 = saved[f'fine_step_{a1}']
        fine_margin_0 = saved[f'fine_margin_{a0}']
        fine_margin_1 = saved[f'fine_margin_{a1}']
        # Update settings with current runtime values
        state['settings'].update({
            'max_generations': max_generations,
            'max_attempts': max_attempts,
            'max_concurrent': max_concurrent,
        })
        save_state(state, run_dir)
    else:
        state = {
            'run_name': run_name,
            'fixed_params': fixed_params,
            'evolved_params': {k: list(v) if isinstance(v, tuple) else v
                             for k, v in evolved_params.items()},
            'problem_name': problem.name,
            'settings': {
                'pop_size': pop_size,
                'max_generations': max_generations,
                'grid_resolution': grid_resolution,
                f'fine_step_{a0}': fine_step_0,
                f'fine_step_{a1}': fine_step_1,
                'coarse_factor': coarse_factor,
                f'fine_margin_{a0}': fine_margin_0,
                f'fine_margin_{a1}': fine_margin_1,
                'acceptance_threshold': acceptance_threshold,
                'max_attempts': max_attempts,
                'max_concurrent': max_concurrent,
                'n_basins': n_basins,
            },
            'candidates': [],
            'generation_history': [],
            'current_gen': 0,
            'best_fitness_history': [],
        }
        candidates_all = []
        gen_start = 0
        best_fitness_history = []

        # ── Initialize population ────────────────────────────────────────
        n_seeds = 0
        if seed_candidates:
            n_seeds = len(seed_candidates)
            for si, seed in enumerate(seed_candidates):
                full = dict(fixed_params)
                full.update(seed)
                candidates_all.append(
                    _make_candidate(f'gen00_cand{si:02d}', seed, full))
            print(f"Seeded {n_seeds} candidate(s) into initial population")
        n_random = pop_size - n_seeds
        print(f"Initializing population: {n_random} random + {n_seeds} seeded "
              f"= {pop_size} candidates")
        population = init_population(evolved_params, n_random, fixed_params,
                                     problem=problem)
        # Re-number random candidates to follow seeds.
        for ri, cand in enumerate(population):
            cand['id'] = f'gen00_cand{n_seeds + ri:02d}'
        candidates_all.extend(population)
        state['candidates'] = candidates_all
        save_state(state, run_dir)

    def _make_sweep_progress(tick_cb, t0):
        """Terminal-only progress (same format as refine, no PNG/CSV)."""
        def on_progress(s):
            if tick_cb and hasattr(tick_cb, 'clear_spinner'):
                tick_cb.clear_spinner()
            elapsed = time.perf_counter() - t0
            # Generic acceptance counting from result maps
            rn0 = problem.result_names[0]
            primary = s.get(f'{rn0}_map', s.get(rn0))
            accept_f = problem.acceptance_field
            accept = s.get(f'{accept_f}_map', s.get(accept_f))
            at = s.get('acceptance_threshold', 0.10)
            converged = ~np.isnan(primary) if primary is not None else np.array([])
            accepted = converged & (accept >= at) if accept is not None else np.array([])
            n_accepted = int(np.sum(accepted))
            n_converged = int(np.sum(converged))
            n_valid = int(np.sum(s['n_attempts_map'] >= 0))
            # Fix: a point accepted on its first attempt has n_attempts==1,
            # which trips `>= max_attempts` when max_attempts==1. Without
            # masking out accepted points, n_remaining goes negative.
            max_attempts_local = s.get('max_attempts', 4)
            attempts_exhausted = s['n_attempts_map'] >= max_attempts_local
            if accepted.shape == attempts_exhausted.shape:
                exhausted = attempts_exhausted & ~accepted
            else:
                exhausted = attempts_exhausted
            n_remaining = n_valid - n_accepted - int(np.sum(exhausted))
            n_timeouts = int(s['timeout_map'].sum())
            print(f"[{elapsed:6.0f}s] "
                  f"accepted={n_accepted}/{n_valid}  "
                  f"converged={n_converged}  "
                  f"remaining={n_remaining}  "
                  f"fails={s['fail_counts']}  "
                  f"timeouts={n_timeouts}  "
                  f"eff={s['effective_max']}  "
                  f"in-flight={s['in_flight']}",
                  flush=True)
        return on_progress

    _shared_tick_cb = make_tick_callback(own_clock=True)

    t_start = time.perf_counter()

    for gen in range(gen_start, max_generations):
        print(f"\n{'='*60}")
        print(f"Generation {gen}")
        print(f"{'='*60}")

        # Get current generation's candidates
        gen_prefix = f'gen{gen:02d}_'
        gen_candidates = [c for c in candidates_all
                          if c['id'].startswith(gen_prefix)]

        # If no candidates for this generation (interrupted between gen++ and breed),
        # breed from previous generation's best candidates
        if not gen_candidates and gen > 0:
            prev_prefix = f'gen{gen-1:02d}_'
            prev_cands = [c for c in candidates_all
                          if c['id'].startswith(prev_prefix)]
            prev_fine = [c for c in prev_cands
                         if c.get('fine_fitness') is not None]
            if prev_fine:
                prev_ranked = sorted(prev_fine,
                    key=lambda c: c['fine_fitness'])
                n_par = max(2, len(prev_ranked) // 2)
                prev_parents = prev_ranked[:n_par]
            else:
                prev_ranked = sorted(
                    [c for c in prev_cands if c.get('coarse_fitness') is not None],
                    key=lambda c: c['coarse_fitness'])
                n_par = max(2, len(prev_ranked) // 2)
                prev_parents = prev_ranked[:n_par]
            print(f"  Recovering: breeding from {len(prev_parents)} "
                  f"gen{gen-1:02d} parents...")
            gen_candidates = breed(prev_parents, evolved_params,
                                   fixed_params, gen, pop_size,
                                   problem=problem)
            candidates_all.extend(gen_candidates)
            state['candidates'] = candidates_all
            save_state(state, run_dir)

        # ── Phase 1: validity pre-filter + coarse scan (parallel) ────────
        import multiprocessing
        total_workers = n_workers or max(1, int(multiprocessing.cpu_count() * 0.5))
        # Each candidate gets the full worker pool — the OS scheduler handles
        # contention. This avoids idle CPUs when one candidate finishes early.
        workers_per_cand = total_workers

        # Pre-filter and prepare candidates
        pending = []
        for cand in gen_candidates:
            if cand['status'] not in ('pending',):
                continue

            # Prepare problem for this candidate's params
            problem.prepare(cand['full_params'])
            vr = problem.valid_range()
            cand['valid_range'] = vr

            if vr is None:
                print(f"  {cand['id']}: no valid range — culled")
                cand['status'] = 'culled'
                cand['coarse_fitness'] = 1e6
                continue

            coarse_step_0 = fine_step_0 * coarse_factor
            coarse_step_1 = fine_step_1 * coarse_factor
            coarse_searched = OrderedDict([
                (a0, make_grid(vr[f'{a0}_min'], vr[f'{a0}_max'], coarse_step_0,
                               grid_resolution)),
                (a1, make_grid(vr[f'{a1}_min'], vr[f'{a1}_max'], coarse_step_1,
                               grid_resolution)),
            ])

            if any(len(a) < 2 for a in coarse_searched.values()):
                print(f"  {cand['id']}: valid range too small — culled")
                cand['status'] = 'culled'
                cand['coarse_fitness'] = 1e6
                continue

            config_hash = problem.config_hash()
            cand['config_hash'] = config_hash
            cand['output_dir'] = os.path.join(
                run_dir, f"{cand['id']}_{config_hash}")

            ax0_arr = coarse_searched[a0]
            ax1_arr = coarse_searched[a1]
            r0 = problem.format_bounds(a0, vr[f'{a0}_min'], vr[f'{a0}_max'])
            r1 = problem.format_bounds(a1, vr[f'{a1}_min'], vr[f'{a1}_max'])
            print(f"  {cand['id']}: valid {r0}, {r1}  "
                  f"({len(ax0_arr)}x{len(ax1_arr)} = "
                  f"{len(ax0_arr)*len(ax1_arr)} pts)")
            pending.append((cand, coarse_searched))

        # Run coarse scans in parallel batches (acceptance=0, coarser resolution, temp dir)
        def _run_coarse(cand, coarse_searched):
            problem.prepare(cand['full_params'])
            coarse_params = problem.coarsen_params(coarse_factor)
            problem.prepare(coarse_params)
            coarse_dir = cand['output_dir'] + '_coarse'
            _shared_tick_cb.clear_spinner()
            print(f"  {cand['id']}: coarse scan starting "
                  f"({workers_per_cand} workers)...")
            on_progress = _make_sweep_progress(_shared_tick_cb, t_start)
            stats = run_sweep(
                coarse_searched, coarse_dir, problem,
                grid_resolution=grid_resolution,
                on_progress=on_progress,
                on_tick=_shared_tick_cb,
                progress_interval=10,
                n_workers=workers_per_cand,
                acceptance_threshold=0,
                max_attempts=max_attempts,
                worker_timeout=30,
                max_timeouts=1,
                export=False,
            )
            return cand, stats

        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_concurrent) as tex:
            futures = {}
            pending_iter = iter(pending)

            # Seed initial batch
            for _ in range(min(max_concurrent, len(pending))):
                try:
                    cand, c_searched = next(pending_iter)
                    futures[tex.submit(_run_coarse, cand, c_searched)] = cand
                except StopIteration:
                    break

            while futures:
                for fut in as_completed(futures):
                    cand = futures.pop(fut)
                    _, stats = fut.result()

                    # Extract top-K basins from coarse results
                    _min_sep = max(2, int(fine_margin_0 /
                            (fine_step_0 * coarse_factor)))
                    basins_raw = problem.fitness_basins(
                        stats, acceptance_threshold=0,
                        n_basins=n_basins, min_separation=_min_sep)
                    if not basins_raw:
                        cand['coarse_fitness'] = 1e6
                        cand['basins'] = []
                        cand['status'] = 'coarse_done'
                        _shared_tick_cb.clear_spinner()
                        print(f"  {cand['id']}: no accepted points — "
                              f"fitness=penalty")
                    else:
                        alpha, best_pt = basins_raw[0]
                        cand['coarse_fitness'] = alpha
                        cand['best_point'] = best_pt
                        cand['basins'] = [
                            {'center': bp,
                             'coarse_fitness': a,
                             'fine_fitness': None,
                             'best_point': None,
                             'output_dir': None,
                             'status': 'pending'}
                            for a, bp in basins_raw
                        ]
                        cand['status'] = 'coarse_done'
                        _shared_tick_cb.clear_spinner()
                        v0 = problem.format_point(a0, best_pt[a0])
                        v1 = problem.format_point(a1, best_pt[a1])
                        print(f"  {cand['id']}: coarse fitness = "
                              f"{problem.format_fitness(alpha)} at {v0}, {v1}"
                              f" ({len(basins_raw)} basin(s))")
                    state['candidates'] = candidates_all
                    with _state_lock:
                        save_state(state, run_dir)

                    # Submit next candidate as slot opens
                    try:
                        cand_next, c_axes = next(pending_iter)
                        futures[tex.submit(_run_coarse, cand_next, c_searched)] = cand_next
                    except StopIteration:
                        pass
                    break  # back to as_completed

        # ── Rank by coarse fitness, select top 50% for fine scan ─────────
        coarse_ranked = sorted(
            [c for c in gen_candidates if c['status'] == 'coarse_done'],
            key=lambda c: c['coarse_fitness'])

        n_survivors = max(2, len(coarse_ranked) // 2)
        survivors = coarse_ranked[:n_survivors]
        for c in coarse_ranked[n_survivors:]:
            c['status'] = 'culled'

        print(f"\nCoarse survivors ({n_survivors}/{len(gen_candidates)}):")
        for c in survivors:
            n_b = len(c.get('basins', []))
            print(f"  {c['id']}: {problem.format_fitness(c['coarse_fitness'])}"
                  f" ({n_b} basin(s))")

        # ── Phase 2: fine scan on survivors (per-basin) ─────────────────
        def _make_basin_fine_axes(basin, vr):
            """Build fine-scan axes zoomed around a basin center."""
            center = basin['center']
            v0_lo = max(center[a0] - fine_margin_0, vr[f'{a0}_min'])
            v0_hi = min(center[a0] + fine_margin_0, vr[f'{a0}_max'])
            v1_lo = max(center[a1] - fine_margin_1, vr[f'{a1}_min'])
            v1_hi = min(center[a1] + fine_margin_1, vr[f'{a1}_max'])
            return OrderedDict([
                (a0, make_grid(v0_lo, v0_hi, fine_step_0, grid_resolution)),
                (a1, make_grid(v1_lo, v1_hi, fine_step_1, grid_resolution)),
            ])

        # Build flat list of (cand, basin_idx, fine_axes) jobs
        fine_pending = []
        for cand in survivors:
            if cand['status'] == 'fine_done':
                continue
            basins = cand.get('basins', [])
            if not basins:
                cand['fine_fitness'] = cand['coarse_fitness']
                cand['status'] = 'fine_done'
                continue
            vr = cand['valid_range']
            for bi, basin in enumerate(basins):
                if basin['status'] == 'fine_done':
                    continue
                fine_searched = _make_basin_fine_axes(basin, vr)
                basin['output_dir'] = cand['output_dir'] + f'_basin{bi}'
                ax0_f, ax1_f = fine_searched[a0], fine_searched[a1]
                _shared_tick_cb.clear_spinner()
                center = basin['center']
                c0_str = problem.format_point(a0, center[a0])
                c1_str = problem.format_point(a1, center[a1])
                print(f"  {cand['id']} basin{bi}: fine scan "
                      f"{len(ax0_f)}x{len(ax1_f)} = "
                      f"{len(ax0_f)*len(ax1_f)} pts "
                      f"(zoom around {c0_str}, {c1_str})")
                fine_pending.append((cand, bi, fine_searched))

        def _run_fine(cand, basin_idx, fine_searched):
            problem.prepare(cand['full_params'])
            basin = cand['basins'][basin_idx]
            _shared_tick_cb.clear_spinner()
            print(f"  {cand['id']} basin{basin_idx}: fine scan starting "
                  f"({workers_per_cand} workers)...")
            on_progress = _make_sweep_progress(_shared_tick_cb, t_start)
            stats = run_sweep(
                fine_searched, basin['output_dir'], problem,
                grid_resolution=grid_resolution,
                on_progress=on_progress,
                on_tick=_shared_tick_cb,
                progress_interval=10,
                n_workers=workers_per_cand,
                acceptance_threshold=acceptance_threshold,
                max_attempts=max_attempts,
                worker_timeout=30,
                max_timeouts=1,
                export=False,
            )
            return cand, basin_idx, stats

        with ThreadPoolExecutor(max_workers=max_concurrent) as tex:
            futures = {}
            fine_iter = iter(fine_pending)

            for _ in range(min(max_concurrent, len(fine_pending))):
                try:
                    cand, bi, f_axes = next(fine_iter)
                    futures[tex.submit(_run_fine, cand, bi, f_axes)] = (cand, bi)
                except StopIteration:
                    break

            while futures:
                for fut in as_completed(futures):
                    cand, bi = futures.pop(fut)
                    _, _, stats = fut.result()
                    basin = cand['basins'][bi]
                    alpha, best_pt = problem.fitness(stats, acceptance_threshold)
                    if alpha is not None:
                        basin['fine_fitness'] = alpha
                        basin['best_point'] = best_pt
                        _shared_tick_cb.clear_spinner()
                        print(f"  {cand['id']} basin{bi}: fine fitness = "
                              f"{problem.format_fitness(alpha)}")
                    else:
                        # No accepted points in the fine scan.  The coarse
                        # scan uses acceptance_threshold=0 to cast a wide
                        # net for basin discovery, so coarse_fitness may
                        # reflect a mode that fails the acceptance filter
                        # (e.g. low coupling efficiency).  Falling back to
                        # coarse_fitness here would let an unacceptable
                        # mode compete in ranking and contaminate breeding.
                        # Assign penalty fitness instead.
                        basin['fine_fitness'] = 1e6
                        basin['best_point'] = basin['center']
                        _shared_tick_cb.clear_spinner()
                        print(f"  {cand['id']} basin{bi}: "
                              f"fine scan no accepted points — penalty")
                    basin['status'] = 'fine_done'

                    # Update candidate top-level from best basin so far
                    done_basins = [b for b in cand['basins']
                                   if b.get('fine_fitness') is not None]
                    if done_basins:
                        best_b = min(done_basins,
                                     key=lambda b: b['fine_fitness'])
                        cand['fine_fitness'] = best_b['fine_fitness']
                        cand['best_point'] = best_b['best_point']

                    # Candidate done when all basins done
                    if all(b['status'] == 'fine_done'
                           for b in cand['basins']):
                        cand['status'] = 'fine_done'

                    state['candidates'] = candidates_all
                    with _state_lock:
                        save_state(state, run_dir)

                    try:
                        cand_next, bi_next, f_searched = next(fine_iter)
                        futures[tex.submit(_run_fine, cand_next, bi_next,
                                           f_searched)] = (cand_next, bi_next)
                    except StopIteration:
                        pass
                    break

        # ── Select top 25% as parents for next generation ────────────────
        fine_ranked = sorted(
            [c for c in survivors if c['status'] == 'fine_done'],
            key=lambda c: c.get('fine_fitness', c['coarse_fitness']))

        if not fine_ranked:
            # No fine results — fall back to coarse ranking
            fine_ranked = coarse_ranked

        n_parents = max(2, len(fine_ranked) // 2)
        parents = fine_ranked[:n_parents]
        if not parents:
            print("No viable parents — stopping")
            break
        best = parents[0]
        best_fitness = best.get('fine_fitness', best['coarse_fitness'])
        best_fitness_history.append(best_fitness)

        gen_info = {
            'generation': gen,
            'best_coarse': min(c['coarse_fitness'] for c in gen_candidates
                               if c['coarse_fitness'] is not None),
            'best_fine': best_fitness,
            'best_id': best['id'],
            'best_outer': best['evolved_values'],
            'best_point': best.get('best_point'),
        }
        state['generation_history'].append(gen_info)
        state['best_fitness_history'] = best_fitness_history
        state['current_gen'] = gen + 1

        elapsed = time.perf_counter() - t_start
        print(f"\nGeneration {gen} complete ({elapsed:.0f}s total)")
        print(f"  Best: {best['id']} = {problem.format_fitness(best_fitness)}")
        print(f"  Params: {best['evolved_values']}")
        if best.get('best_point'):
            bp = best['best_point']
            print(f"  At {problem.format_point(a0, bp[a0])}, "
                  f"{problem.format_point(a1, bp[a1])}")

        save_state(state, run_dir)
        save_summary(state, run_dir)

        if on_generation:
            on_generation(state)

        # ── Convergence check ────────────────────────────────────────────
        if len(best_fitness_history) >= convergence_patience + 1:
            recent = best_fitness_history[-convergence_patience:]
            old_best = best_fitness_history[-(convergence_patience + 1)]
            improvement = (old_best - min(recent)) / max(abs(old_best), 1e-12)
            if improvement < convergence_threshold:
                print(f"\nConverged: <{convergence_threshold*100:.0f}% improvement "
                      f"over {convergence_patience} generations")
                break

        # ── Breed next generation ────────────────────────────────────────
        if gen + 1 < max_generations:
            # Elitism: carry forward the top 1-2 parents unchanged
            n_elite = min(2, len(parents))
            # Immigrants: fresh random candidates for diversity
            n_immigrant = max(1, pop_size // 4)
            n_bred = pop_size - n_elite - n_immigrant

            next_gen = gen + 1
            new_candidates = []

            # Elite (copy parents with new IDs)
            for ei in range(n_elite):
                elite = _make_candidate(
                    f'gen{next_gen:02d}_cand{ei:02d}',
                    dict(parents[ei]['evolved_values']),
                    dict(parents[ei]['full_params']),
                    parents=[parents[ei]['id']],
                )
                new_candidates.append(elite)

            # Bred children
            bred = breed(parents, evolved_params, fixed_params,
                         next_gen, n_bred,
                         start_idx=n_elite, problem=problem)
            new_candidates.extend(bred)

            # Immigrants (fresh random)
            immigrants = init_population(evolved_params, n_immigrant, fixed_params,
                                        problem=problem)
            for ii, imm in enumerate(immigrants):
                imm['id'] = f'gen{next_gen:02d}_cand{n_elite + n_bred + ii:02d}'
                new_candidates.append(imm)

            print(f"\nGeneration {next_gen}: {n_elite} elite + "
                  f"{n_bred} bred + {n_immigrant} immigrants = "
                  f"{len(new_candidates)} candidates")

            candidates_all.extend(new_candidates)
            state['candidates'] = candidates_all
            with _state_lock:
                        save_state(state, run_dir)

    # ── Final summary ────────────────────────────────────────────────────
    all_done = [c for c in candidates_all
                if c.get('fine_fitness') is not None and c['status'] == 'fine_done']
    if all_done:
        overall_best = min(all_done,
                           key=lambda c: c.get('fine_fitness', 1e6))
        print(f"\n{'='*60}")
        print(f"Optimization complete: {len(state['generation_history'])} generations")
        print(f"Best: {overall_best['id']} = "
              f"{problem.format_fitness(overall_best['fine_fitness'])}")
        print(f"Outer params: {overall_best['evolved_values']}")
        if overall_best.get('best_point'):
            bp = overall_best['best_point']
            print(f"Best {problem.format_point(a0, bp[a0])}, "
                  f"{problem.format_point(a1, bp[a1])}")
        print(f"Output: {overall_best['output_dir']}")

    return state
