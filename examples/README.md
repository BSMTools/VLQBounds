# Examples

Run from the repository root after `python -m pip install -e ".[plot]"`.

## Parameter scans

Each script performs a 2D scan in mass and mixing, writes a CSV file and an
exclusion plot.

| Script | Representation |
|---|---|
| `scan_t_singlet.py` | $T$ singlet |
| `scan_b_singlet.py` | $B$ singlet |
| `scan_t_doublet_tb.py`, `scan_b_doublet_tb.py` | $(T,B)$ doublet |
| `scan_t_doublet_xt.py`, `scan_x_doublet_xt.py` | $(X,T)$ doublet |
| `scan_b_doublet_by.py` | $(B,Y)$ doublet |

```bash
python examples/scan_t_singlet.py
```

## Summary figures

```bash
python examples/plot_limit_landscape.py                    # T and B singlets -> docs/figures/
python examples/plot_limit_landscape.py --species B --n-mass 49 --n-mix 40   # faster preview
```

## Validation

```bash
python examples/validate_imports.py
python examples/validate_runtime_smoke.py
python examples/validate_example_suite.py
python examples/validate_new_coupling_limits.py            # report + plot in build/validation/
python -m pytest                                           # physics regression tests
```
