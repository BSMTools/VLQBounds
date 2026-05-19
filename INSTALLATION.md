# Installation

## 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

## 2. Upgrade packaging tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 3. Install the package

Standard installation:

```bash
pip install .
```

Editable installation for development:

```bash
pip install -e .
```


## 4. Quick import test

```bash
python -c "from vlqbounds.inference import VLQBounds; print('VLQBounds import OK')"
```
