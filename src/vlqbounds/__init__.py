"""VLQBounds: confront vector-like quark models with ATLAS and CMS exclusion limits."""
from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("vlqbounds")
except PackageNotFoundError:  # running from a source checkout without installation
    __version__ = "1.1.0"

__all__ = ["__version__"]
