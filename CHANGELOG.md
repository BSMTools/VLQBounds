# Changelog

All notable changes to VLQBounds are documented here. The project follows
[Semantic Versioning](https://semver.org/).

## [1.1.0] — 2026-11

### Added
- Single-production coupling limits
  - CMS [arXiv:2606.01423](https://arxiv.org/abs/2606.01423), Fig. 10 — singlet $B$ via $bqB$ and $tqB$, $B\to tW$
  - ATLAS [arXiv:2506.15515](https://arxiv.org/abs/2506.15515), Fig. 6a — singlet $T$, $T\to W(\ell\nu)b$
  - CMS [arXiv:2604.17564](https://arxiv.org/abs/2604.17564), Fig. 4 — singlet $T$, $T\to Wb$
- `CouplingLimitDescriptor.limit_scale`: maps experimental coupling conventions
  onto the package convention $\kappa = \sqrt{2}\,\sin\theta_L$ (singlets).
- Physics regression tests (`tests/`) reproducing numbers quoted in the papers.
- `examples/plot_limit_landscape.py`: publication-quality summary figures.
- `examples/validate_new_coupling_limits.py`: full validation report and plot.
- Continuous integration (Python 3.10–3.13), `CITATION.cff`, `LICENSE`,
  `CONTRIBUTING.md`.
- `VLQBOUNDS_DATA` environment variable to point at a custom data directory.

### Fixed
- **Observed/expected order of cross-section limits.** Tables are stored as
  `[mass, observed, expected]` but were read as `[mass, expected, observed]`,
  so every pair- and single-production cross-section verdict used the
  expected limit as the observed one. The singlet-$T$ observed mass limit
  moves from 1410 to 1485 GeV.
- Switching parameterisation (`s_l` → `k_*` → `w_m`) on the same `VLQBounds`
  object no longer keeps the previous mixing, coupling or width: each
  `*_params(...)` call now fully defines the point.
- The data directory is found from any working directory for editable installs.
- `examples/validate_imports.py` imports `ExperimentalBoundsRepository` from
  `vlqbounds.constraints`.
- Version and author list in the console banner and package metadata.

### Removed
- Committed virtual environment, byte-code, `egg-info` and generated run
  manifests from version control.

## [1.0] — 2026

- Initial release of the restructured package (`vlqbounds.theory`,
  `vlqbounds.models`, `vlqbounds.inference`, `vlqbounds.constraints`).
