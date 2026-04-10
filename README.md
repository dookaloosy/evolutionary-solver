# evolutionary-solver

Domain-agnostic grid sweep + evolutionary optimizer with continuation seeding.

Designed for expensive objective functions with spatial structure — neighboring grid points seed each other's solvers, enabling efficient exploration of eigenvalue/PDE parameter spaces.

## Install

```bash
pip install -e .
```

## Usage

Implement the `Problem` protocol for your domain, then:

```python
from sweep_engine import run_sweep, Problem
from optimizer_engine import run_optimizer

class MyProblem(Problem):
    name = 'my_problem'
    result_names = ('eigenvalue', 'residual', 'quality')
    acceptance_field = 'quality'

    def prepare(self, params):
        ...
    def evaluate(self, axis_values, seed, timeout):
        ...
    def axis_names(self):
        return ['ax0', 'ax1']
    # ... other required methods (see sweep_engine.Problem)

problem = MyProblem()
problem.prepare(fixed_params)

# Grid sweep — grid_resolution is required (snap step for axis values)
stats = run_sweep(
    searched_axes, output_dir, problem,
    grid_resolution=1e-6,
)

# Evolutionary optimizer — caller must supply grid + step parameters
state = run_optimizer(
    fixed_params, evolved_params,
    grid_resolution=1e-6,
    fine_step_0=25e-6, fine_step_1=10e-6,
    fine_margin_0=100e-6, fine_margin_1=40e-6,
    problem=problem,
)
```

The engine ships no domain-specific defaults — every scale-dependent value
(`grid_resolution`, `fine_step_*`, `fine_margin_*`) is a required parameter
of the calling code. Pick values that match your problem's units.

## Architecture

- `sweep_engine.py` — N-dimensional grid sweep with continuation seeding, `Problem` base class
- `optimizer_engine.py` — Evolutionary search over outer parameters, coarse-to-fine multi-resolution
