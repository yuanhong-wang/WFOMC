"""
Adapter for the `ganak` model counter (https://github.com/meelgroup/ganak).

It writes a weighted DIMACS-like file in the Model Counting Competition
format and invokes ganak as an external process:

* rational weighted counting via ``--mode 1`` (output ``c s exact arb frac``);
* multivariate-polynomial (symbolic) weighted counting via
  ``--mode 3 --npolyvars N`` (output ``c s exact poly``).

Note: the polynomial mode only prints its result on the ``devel`` branch of
ganak, so a build from that branch is required for symbolic weights.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

# WFOMC counts can have many thousands of digits. Lift Python 3.11+'s default
# 4300-digit limit on ``int(str)`` so that parsing ganak output doesn't fail
# on large numerators / denominators. ``0`` disables the limit.
try:  # pragma: no cover -- defensive on older Pythons
    sys.set_int_max_str_digits(0)
except (AttributeError, ValueError):
    pass

from loguru import logger
from flint import fmpq, fmpq_mpoly, fmpq_mpoly_ctx

from wfomc.utils import Rational, RingElement


GANAK_ENV_VAR: str = 'GANAK'
_FRAC_PREFIX: str = 'c s exact arb frac '
_POLY_PREFIX: str = 'c s exact poly '
_BUILD_HINT: str = (
    "ganak binary not found. Build it from the `devel` branch of "
    "https://github.com/meelgroup/ganak (the polynomial mode only emits "
    "output on that branch), put it on PATH, or set the "
    f"{GANAK_ENV_VAR} environment variable to the binary path."
)


class GanakError(RuntimeError):
    """Raised when ganak is missing, fails, or returns unparsable output."""


def find_ganak(ganak_path: str = None) -> str:
    """Locate the ganak binary.

    Search order: explicit ``ganak_path`` argument, the ``GANAK`` environment
    variable, then ``ganak`` on ``PATH``.

    Args:
        ganak_path: Explicit path to the ganak binary, or None.

    Returns:
        Absolute path to an executable ganak binary.

    Raises:
        GanakError: When no usable binary can be found.
    """
    candidates = [ganak_path, os.environ.get(GANAK_ENV_VAR)]
    for candidate in candidates:
        if not candidate:
            continue
        resolved = shutil.which(candidate) or candidate
        if os.path.isfile(resolved) and os.access(resolved, os.X_OK):
            return resolved
        raise GanakError(f'ganak binary not found or not executable: {candidate}')
    found = shutil.which('ganak')
    if found is not None:
        return found
    raise GanakError(_BUILD_HINT)


def _weight_to_str(weight: RingElement) -> str:
    """Serialize a weight (rational or polynomial) for a ganak weight line.

    Spaces are stripped so the value is a single token; FLINT's pretty
    printer (used by both this project and ganak) round-trips losslessly.
    """
    return str(weight).replace(' ', '')


def _make_aux_ctx(npolyvars: int) -> fmpq_mpoly_ctx:
    """Build a FLINT context whose generators match ganak's polynomial vars.

    ganak's mode-3 polynomial parser uses zero-indexed variables ``x0,
    x1, ..., x_{N-1}`` when invoked with ``--npolyvars N``; any other name
    (including the semantic ones cofola emits, like ``v_payment_nickel``)
    is rejected with "Unexpected char". Round-tripping every polynomial
    through this aux context renames variables to ``xK`` for ganak and
    back to the caller's names after parsing.
    """
    return fmpq_mpoly_ctx.get(tuple(f'x{i}' for i in range(npolyvars)))


def _remap_poly(poly: RingElement, target_ctx: fmpq_mpoly_ctx) -> RingElement:
    """Rebuild an ``fmpq_mpoly`` in ``target_ctx`` preserving its monomials.

    Plain rationals and integers pass through unchanged. The source and
    target contexts must have the same arity; only the generator names
    differ.
    """
    if not isinstance(poly, fmpq_mpoly):
        return poly
    return target_ctx.from_dict(dict(zip(poly.monoms(), poly.coeffs())))


def _build_dimacs(n_vars: int,
                   clauses: Iterable[Iterable[int]],
                   weights: dict[int, tuple[RingElement, RingElement]],
                   aux_ctx: fmpq_mpoly_ctx | None = None) -> str:
    """Build the weighted DIMACS-like text fed to ganak.

    When ``aux_ctx`` is provided every polynomial weight is rewritten in
    that context so ganak sees only ``xN`` variable names.
    """
    clause_lines = [
        ' '.join(str(lit) for lit in clause) + ' 0' for clause in clauses
    ]
    lines = ['c t wmc', f'p cnf {n_vars} {len(clause_lines)}']
    for var in sorted(weights):
        pos, neg = weights[var]
        if aux_ctx is not None:
            pos = _remap_poly(pos, aux_ctx)
            neg = _remap_poly(neg, aux_ctx)
        lines.append(f'c p weight {var} {_weight_to_str(pos)} 0')
        lines.append(f'c p weight -{var} {_weight_to_str(neg)} 0')
    lines.extend(clause_lines)
    return '\n'.join(lines) + '\n'


def _parse_frac(token: str) -> Rational:
    """Parse a ``p/q`` (or ``p``) rational token into a flint ``fmpq``."""
    token = token.strip()
    if '/' in token:
        num, den = token.split('/', 1)
        return fmpq(int(num), int(den))
    return fmpq(int(token))


def _parse_output(stdout: str, symbolic: bool,
                  poly_ctx: fmpq_mpoly_ctx,
                  aux_ctx: fmpq_mpoly_ctx | None = None) -> RingElement:
    """Extract the model count from ganak's stdout.

    When ``aux_ctx`` is provided the polynomial result is parsed in it
    (ganak emits its output using the renamed ``xN`` generators) and then
    rebuilt in ``poly_ctx`` so the caller's variable names survive.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if not symbolic and line.startswith(_FRAC_PREFIX):
            return _parse_frac(line[len(_FRAC_PREFIX):])
        if symbolic and line.startswith(_POLY_PREFIX):
            text = line[len(_POLY_PREFIX):].strip()
            if aux_ctx is not None:
                aux_poly = fmpq_mpoly(text, aux_ctx)
                return _remap_poly(aux_poly, poly_ctx)
            return fmpq_mpoly(text, poly_ctx)
    raise GanakError(
        'could not find an exact count in ganak output; '
        'is this a ganak build from the `devel` branch?\n' + stdout
    )


def ganak_count(n_vars: int,
                clauses: Iterable[Iterable[int]],
                weights: dict[int, tuple[RingElement, RingElement]],
                symbolic: bool = False,
                npolyvars: int = 0,
                poly_ctx: fmpq_mpoly_ctx = None,
                ganak_path: str = None,
                timeout: float = None) -> RingElement:
    """Run ganak on a weighted CNF and return the weighted model count.

    Args:
        n_vars: Number of propositional variables.
        clauses: Iterable of clauses, each an iterable of signed ints.
        weights: Map from variable id to its (positive, negative) literal
            weights.
        symbolic: When True use the polynomial mode (``--mode 3``); otherwise
            the exact rational mode (``--mode 1``).
        npolyvars: Number of polynomial variables (required when symbolic).
        poly_ctx: FLINT context used to parse the polynomial result; its
            generators must match the symbolic weight variables.
        ganak_path: Explicit path to the ganak binary, or None to auto-detect.
        timeout: Optional subprocess timeout in seconds.

    Returns:
        The weighted model count as a rational or a polynomial.

    Raises:
        GanakError: When ganak is missing, fails, or returns bad output.
    """
    binary = find_ganak(ganak_path)
    clauses = [list(clause) for clause in clauses]

    aux_ctx: fmpq_mpoly_ctx | None = None
    if symbolic:
        if npolyvars <= 0 or poly_ctx is None:
            raise GanakError('symbolic counting requires npolyvars > 0 and a '
                              'polynomial context')
        # ganak's polynomial parser only accepts xN-style variable names,
        # while callers (e.g. cofola) use semantic names like
        # ``v_payment_nickel``. Round-trip through an aux context with
        # generators ``x1, ..., xN`` to keep ganak happy without forcing
        # the caller to rename anything.
        aux_ctx = _make_aux_ctx(npolyvars)

    content = _build_dimacs(n_vars, clauses, weights, aux_ctx)

    cmd = [binary, '--prob', '0']
    if symbolic:
        cmd += ['--mode', '3', '--npolyvars', str(npolyvars)]
    else:
        cmd += ['--mode', '1']

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.cnf', prefix='wfomc_ganak_', delete=False
    )
    try:
        tmp.write(content)
        tmp.close()
        cmd.append(tmp.name)
        logger.debug('Invoking ganak: {}', ' '.join(cmd))
        logger.debug('ganak input ({} vars, {} clauses):\n{}',
                     n_vars, len(clauses), content)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as e:
            raise GanakError(f'ganak timed out after {timeout}s') from e
        except OSError as e:
            raise GanakError(f'failed to run ganak: {e}') from e
        if proc.returncode != 0:
            raise GanakError(
                f'ganak exited with code {proc.returncode}\n'
                f'stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}'
            )
        return _parse_output(proc.stdout, symbolic, poly_ctx, aux_ctx)
    finally:
        os.unlink(tmp.name)
