# Contributing to VLQBounds

Contributions of new experimental results, bug fixes and documentation are
welcome. Please open an issue or a pull request on
[GitHub](https://github.com/BSMTools/VLQBounds).

## Development setup

```bash
git clone https://github.com/BSMTools/VLQBounds.git
cd VLQBounds
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest
```

## Adding a new experimental limit

1. **Digitise or download the curve** (HEPData whenever available) and store it
   as JSON under `data/<species>/<ATLAS|CMS|Theo>_Tables/`:

   ```json
   {
     "n_columns": 3,
     "rows": [[800.0, 0.306, 0.249], [900.0, 0.177, 0.232]],
     "columns": ["Mass(GeV)", "obs(kappa_CMS)", "exp(kappa_CMS)"]
   }
   ```

   Rows are always `[mass, observed, expected, ...]`. Keep the values exactly as
   published, in the experiment's own convention.

2. **Register it.**
   - Cross-section limits: `src/vlqbounds/constraints/experimental_repository.py`
     (and, for single production, the theory prediction in `src/vlqbounds/models/`).
   - Coupling limits: `_RAW_COUPLING_LIMIT_REGISTRY` in
     `src/vlqbounds/inference/coupling_limits.py`.

3. **Fix the coupling convention.** VLQBounds uses
   $\kappa_T=\kappa_B=\sqrt{2}\,\sin\theta_L$ for singlets. Read the paper's
   Lagrangian and set `limit_scale` accordingly:

   | Experimental coupling | Lagrangian | `limit_scale` |
   |---|---|---|
   | $\kappa\approx\sin\theta$ | $\frac{g}{\sqrt2}\,\kappa\,\bar Q\gamma^\mu W_\mu q$ | $\sqrt2$ |
   | $\kappa_W$ with range $[0,\sqrt2]$ for singlets | $\frac{g}{2}\,\kappa_W\,\bar Q\gamma^\mu W_\mu q$ | $1$ |

4. **Add a regression test** in `tests/test_physics_benchmarks.py` that
   reproduces a number quoted in the paper (a mass limit at a benchmark
   coupling or width, or a coupling limit at a given mass).

5. Run `python -m pytest` and `python examples/plot_limit_landscape.py`, and
   add an entry to `CHANGELOG.md`.

## Style

- Python ≥ 3.10, type hints where practical, no new runtime dependencies
  without discussion.
- Keep commits focused; describe the physics change and how it was validated.
