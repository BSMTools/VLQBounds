from pathlib import Path

REQUIRED_EXAMPLES = [
    "scan_b_singlet.py",
    "scan_b_doublet_by.py",
    "scan_b_doublet_tb.py",
    "scan_t_singlet.py",
    "scan_t_doublet_tb.py",
    "scan_t_doublet_xt.py",
    "scan_x_doublet_xt.py",
    "example_support.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    missing = [name for name in REQUIRED_EXAMPLES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing example files: {missing}")
    print("Example suite manifest validation passed.")


if __name__ == "__main__":
    main()
