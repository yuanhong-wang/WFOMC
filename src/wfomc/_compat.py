import os

# Set WFOMC_CYTHON_ONLY=1 to raise loudly when Cython .so is missing.
# Default: silently fall back to pure Python.
CYTHON_ONLY = os.environ.get("WFOMC_CYTHON_ONLY", "0") == "1"


def try_import_cython(cython_mod: str, python_fallback: str):
    """Import Cython extension if compiled; fall back to Python module.

    Args:
        cython_mod: Fully-qualified module name for the Cython .so.
        python_fallback: Fully-qualified module name for the Python .py.

    Returns:
        The imported module (Cython if available, else Python).

    Raises:
        ImportError: If Cython .so is missing and WFOMC_CYTHON_ONLY=1.
    """
    try:
        return __import__(cython_mod, fromlist=["*"])
    except ImportError:
        if CYTHON_ONLY:
            raise ImportError(
                f"Cython module '{cython_mod}' is not compiled. "
                "Run: python setup.py build_ext --inplace\n"
                "Or unset WFOMC_CYTHON_ONLY to use the Python fallback."
            )
        return __import__(python_fallback, fromlist=["*"])
