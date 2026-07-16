"""Shared adapter for the optional Ganak exact weighted model counter."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from time import perf_counter
from typing import TypeAlias

from flint import fmpq, fmpq_mpoly, fmpq_mpoly_ctx

from wfomc.errors import GanakError

ExactGanakValue: TypeAlias = fmpq | fmpq_mpoly
logger = logging.getLogger(__name__)


# WFOMC counts can have many thousands of digits. Lift Python 3.11+'s default
# 4300-digit limit on ``int(str)`` so that parsing Ganak output does not fail
# on large numerators or denominators. ``0`` disables the limit.
try:  # pragma: no cover -- defensive on older Pythons
    sys.set_int_max_str_digits(0)
except (AttributeError, ValueError):
    pass


GANAK_REPO_URL: str = "https://github.com/meelgroup/ganak.git"
GANAK_COMMIT: str = "82a1d1fb6f0d6fb4a46b825f84b29567728ae483"
GANAK_ARJUN_REPO_URL: str = "https://github.com/meelgroup/arjun.git"
GANAK_ARJUN_COMMIT: str = "1553e6b3ebdd76ba3b66d3fece4cf8de4e2743ce"
GANAK_ENV_VAR: str = "GANAK"
_FRAC_PREFIX: str = "c s exact arb frac "
_POLY_PREFIX: str = "c s exact poly "
_BUILD_HINT: str = (
    "ganak binary not found. Run `uv run wfomc-install-ganak` to build the "
    f"pinned ganak binary ({GANAK_COMMIT}) into the uv environment, put "
    f"`ganak` on PATH, or set the {GANAK_ENV_VAR} environment variable to "
    "the binary path."
)


def find_ganak(ganak_path: str | None = None) -> str:
    """Locate an executable Ganak binary."""

    candidates = [ganak_path, os.environ.get(GANAK_ENV_VAR)]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) or candidate
        if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved
        raise GanakError(f"ganak binary not found or not executable: {candidate}")
    found = shutil.which("ganak")
    if found is not None:
        return found
    raise GanakError(_BUILD_HINT)


def _weight_to_str(weight: ExactGanakValue) -> str:
    """Serialize a rational or polynomial weight as one Ganak token."""

    return str(weight).replace(" ", "")


def _make_aux_ctx(npolyvars: int) -> fmpq_mpoly_ctx:
    """Build a FLINT context whose generators match Ganak's polynomial vars."""

    return fmpq_mpoly_ctx.get(tuple(f"x{i}" for i in range(npolyvars)))


def _remap_poly(
    poly: ExactGanakValue,
    target_ctx: fmpq_mpoly_ctx,
) -> ExactGanakValue:
    """Rebuild an ``fmpq_mpoly`` in ``target_ctx`` preserving monomials."""

    if not isinstance(poly, fmpq_mpoly):
        return poly
    return target_ctx.from_dict(dict(zip(poly.monoms(), poly.coeffs())))


def _build_dimacs(
    n_vars: int,
    clauses: Iterable[Iterable[int]],
    weights: dict[int, tuple[ExactGanakValue, ExactGanakValue]],
    aux_ctx: fmpq_mpoly_ctx | None = None,
) -> str:
    """Build the weighted DIMACS-like text fed to Ganak."""

    clause_lines = [" ".join(str(lit) for lit in clause) + " 0" for clause in clauses]
    lines = ["c t wmc", f"p cnf {n_vars} {len(clause_lines)}"]
    for var in sorted(weights):
        pos, neg = weights[var]
        if aux_ctx is not None:
            pos = _remap_poly(pos, aux_ctx)
            neg = _remap_poly(neg, aux_ctx)
        lines.append(f"c p weight {var} {_weight_to_str(pos)} 0")
        lines.append(f"c p weight -{var} {_weight_to_str(neg)} 0")
    lines.extend(clause_lines)
    return "\n".join(lines) + "\n"


def _parse_frac(token: str) -> fmpq:
    """Parse a ``p/q`` or integer token into a FLINT rational."""

    token = token.strip()
    if "/" in token:
        num, den = token.split("/", 1)
        return fmpq(int(num), int(den))
    return fmpq(int(token))


def _parse_output(
    stdout: str,
    symbolic: bool,
    poly_ctx: fmpq_mpoly_ctx,
    aux_ctx: fmpq_mpoly_ctx | None = None,
) -> ExactGanakValue:
    """Extract the exact model count from Ganak's stdout."""

    for line in stdout.splitlines():
        line = line.strip()
        if not symbolic and line.startswith(_FRAC_PREFIX):
            return _parse_frac(line[len(_FRAC_PREFIX) :])
        if symbolic and line.startswith(_POLY_PREFIX):
            text = line[len(_POLY_PREFIX) :].strip()
            if aux_ctx is not None:
                aux_poly = fmpq_mpoly(text, aux_ctx)
                return _remap_poly(aux_poly, poly_ctx)
            return fmpq_mpoly(text, poly_ctx)
    raise GanakError(
        "could not find an exact count in Ganak output; "
        "is this a Ganak build from the `devel` branch?\n" + stdout
    )


def ganak_count(
    n_vars: int,
    clauses: Iterable[Iterable[int]],
    weights: dict[int, tuple[ExactGanakValue, ExactGanakValue]],
    symbolic: bool = False,
    npolyvars: int = 0,
    poly_ctx: fmpq_mpoly_ctx | None = None,
    ganak_path: str | None = None,
    timeout: float | None = None,
) -> ExactGanakValue:
    """Run Ganak on a weighted CNF and return its exact weighted count."""

    if n_vars == 0:
        # Ganak asserts on empty input, so return the identity directly.
        return fmpq(1)

    binary = find_ganak(ganak_path)
    clauses = [list(clause) for clause in clauses]

    aux_ctx: fmpq_mpoly_ctx | None = None
    if symbolic:
        if npolyvars <= 0 or poly_ctx is None:
            raise GanakError(
                "symbolic counting requires npolyvars > 0 and a polynomial context"
            )
        aux_ctx = _make_aux_ctx(npolyvars)

    content = _build_dimacs(n_vars, clauses, weights, aux_ctx)

    cmd = [binary, "--prob", "0"]
    if symbolic:
        cmd += ["--mode", "3", "--npolyvars", str(npolyvars)]
    else:
        cmd += ["--mode", "1"]

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".cnf", prefix="wfomc_ganak_", delete=False
    )
    try:
        tmp.write(content)
        tmp.close()
        cmd.append(tmp.name)
        logger.debug("Invoking Ganak: %s", " ".join(cmd))
        started = perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise GanakError(f"Ganak timed out after {timeout}s") from exc
        except OSError as exc:
            raise GanakError(f"failed to run Ganak: {exc}") from exc
        if proc.returncode != 0:
            raise GanakError(
                f"Ganak exited with code {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        logger.info(
            "Ganak completed: vars=%d clauses=%d elapsed_ms=%.3f",
            n_vars,
            len(clauses),
            (perf_counter() - started) * 1000,
        )
        return _parse_output(proc.stdout, symbolic, poly_ctx, aux_ctx)
    finally:
        os.unlink(tmp.name)


__all__ = [
    "GANAK_ARJUN_COMMIT",
    "GANAK_ARJUN_REPO_URL",
    "GANAK_COMMIT",
    "GANAK_ENV_VAR",
    "GANAK_REPO_URL",
    "GanakError",
    "find_ganak",
    "ganak_count",
]
