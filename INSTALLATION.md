# Installation

## 1. Get the code

```bash
git clone https://github.com/BSMTools/VLQBounds.git
cd VLQBounds
```

## 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## 3. Install

```bash
python -m pip install -e .            # core: numpy, scipy, pandas
python -m pip install -e ".[plot]"    # + matplotlib, for the figure scripts
python -m pip install -e ".[dev]"     # + pytest, for development
```

The limit tables live in `data/` at the repository root and are found
automatically for editable installs, from any working directory. To use a
different copy of the tables, set

```bash
export VLQBOUNDS_DATA=/path/to/data
```

## 4. Check the installation

```bash
python -c "import vlqbounds; print('VLQBounds', vlqbounds.__version__)"
python examples/validate_runtime_smoke.py
python -m pytest                      # physics regression tests (needs the [dev] extra)
```
