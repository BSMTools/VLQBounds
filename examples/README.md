# Updated VLQBounds Examples

These examples use the **new package modules** directly:

- `vlqbounds.models`
- `vlqbounds.inference`
- `vlqbounds.theory`

## Main scan examples

- `scan_b_singlet.py`
- `scan_b_doublet_by.py`
- `scan_b_doublet_tb.py`
- `scan_t_singlet.py`
- `scan_t_doublet_tb.py`
- `scan_t_doublet_xt.py`
- `scan_x_doublet_xt.py`

Each script performs a 2D scan in mass and mixing, writes a CSV file, and produces a PDF exclusion plot.

## Validation scripts

- `validate_imports.py`
- `validate_runtime_smoke.py`
- `validate_example_suite.py`

## Run a scan

```bash
python scan_t_singlet.py
```

## Run validations

```bash
python validate_imports.py
python validate_runtime_smoke.py
python validate_example_suite.py
```
