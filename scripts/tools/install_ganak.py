from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from wfomc.algo.ganak import GANAK_COMMIT, GANAK_REPO_URL


def _run(
    cmd: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    printable = " ".join(cmd)
    where = f" (cwd={cwd})" if cwd is not None else ""
    print(f"+ {printable}{where}")
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _default_install_dir() -> Path:
    return Path(sys.prefix) / ("Scripts" if os.name == "nt" else "bin")


def _binary_name() -> str:
    return "ganak.exe" if os.name == "nt" else "ganak"


def _require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"required build tool not found on PATH: {name}")


def _brew_prefixes(packages: tuple[str, ...]) -> list[Path]:
    if shutil.which("brew") is None:
        return []

    prefixes: list[Path] = []
    for package in packages:
        try:
            completed = subprocess.run(
                ["brew", "--prefix", package],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            continue

        prefix = Path(completed.stdout.strip())
        if prefix.exists():
            prefixes.append(prefix)

    return prefixes


def _append_env_path(env: dict[str, str], name: str, paths: list[Path]) -> None:
    values = [str(path) for path in paths if path.exists()]
    if not values:
        return
    existing = env.get(name)
    if existing:
        values.append(existing)
    env[name] = os.pathsep.join(values)


def _dependency_hints() -> tuple[list[str], dict[str, str]]:
    cmake_args: list[str] = []
    env = os.environ.copy()

    if sys.platform == "darwin":
        arch = platform.machine()
        if arch:
            cmake_args.append(f"-DCMAKE_OSX_ARCHITECTURES={arch}")
        cmake_args += [
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            "-DCMAKE_INSTALL_RPATH=@loader_path",
        ]
    elif sys.platform.startswith("linux"):
        cmake_args += [
            "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
            "-DCMAKE_INSTALL_RPATH=$ORIGIN",
        ]

    prefixes = _brew_prefixes(("gmp", "mpfr", "flint"))
    if prefixes:
        cmake_args.append(
            "-DCMAKE_PREFIX_PATH=" + ";".join(str(prefix) for prefix in prefixes)
        )
        _append_env_path(env, "CPATH", [prefix / "include" for prefix in prefixes])
        _append_env_path(env, "LIBRARY_PATH", [prefix / "lib" for prefix in prefixes])

    return cmake_args, env


def _shared_library_globs() -> tuple[str, ...]:
    if sys.platform == "darwin":
        return ("*.dylib",)
    if os.name == "nt":
        return ("*.dll",)
    return ("*.so", "*.so.*")


def _copy_shared_libraries(build: Path, install_dir: Path) -> list[Path]:
    lib_dirs = [build / "lib", *(build / "_deps").glob("*-build/lib")]
    copied: list[Path] = []

    for lib_dir in lib_dirs:
        if not lib_dir.exists():
            continue
        for glob in _shared_library_globs():
            for library in sorted(lib_dir.glob(glob)):
                target = install_dir / library.name
                shutil.copy2(library, target, follow_symlinks=True)
                copied.append(target)

    return copied


def _add_darwin_rpath(path: Path) -> None:
    if sys.platform != "darwin" or shutil.which("install_name_tool") is None:
        return

    try:
        subprocess.run(
            ["install_name_tool", "-add_rpath", "@loader_path", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        if "would duplicate path" not in exc.stderr:
            raise


def install_ganak(
    *,
    repo_url: str = GANAK_REPO_URL,
    commit: str = GANAK_COMMIT,
    install_dir: Path | None = None,
    jobs: int | None = None,
    force: bool = False,
) -> Path:
    """Build pinned ganak and install it into the active environment."""

    for tool in ("git", "cmake"):
        _require_tool(tool)

    install_dir = install_dir or _default_install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)
    target = install_dir / _binary_name()

    if target.exists() and not force:
        print(f"ganak already exists at {target}; use --force to rebuild")
        return target

    with tempfile.TemporaryDirectory(prefix="wfomc_ganak_build_") as tmp:
        root = Path(tmp)
        src = root / "ganak"
        build = src / "build"

        _run(["git", "clone", "--recurse-submodules", repo_url, str(src)])
        _run(["git", "checkout", commit], cwd=src)
        _run(["git", "submodule", "update", "--init", "--recursive"], cwd=src)
        dependency_args, build_env = _dependency_hints()
        _run([
            "cmake",
            "-S",
            str(src),
            "-B",
            str(build),
            "-DBUILD_SHARED_LIBS=ON",
            *dependency_args,
        ], env=build_env)

        build_cmd = ["cmake", "--build", str(build), "--target", "ganak-bin"]
        if jobs is not None:
            build_cmd += ["--parallel", str(jobs)]
        else:
            build_cmd += ["--parallel"]
        _run(build_cmd, env=build_env)

        built = build / _binary_name()
        if not built.exists():
            matches = sorted(build.rglob(_binary_name()))
            if not matches:
                raise SystemExit("ganak build completed but binary was not found")
            built = matches[0]

        shutil.copy2(built, target)
        shared_libraries = _copy_shared_libraries(build, install_dir)
        target.chmod(target.stat().st_mode | 0o111)
        _add_darwin_rpath(target)
        for library in shared_libraries:
            _add_darwin_rpath(library)

    print(f"installed ganak {commit} to {target}")
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the pinned ganak binary used by WFOMC.",
    )
    parser.add_argument(
        "--repo",
        default=GANAK_REPO_URL,
        help=f"Ganak git repository URL. Default: {GANAK_REPO_URL}",
    )
    parser.add_argument(
        "--commit",
        default=GANAK_COMMIT,
        help=f"Ganak commit to build. Default: {GANAK_COMMIT}",
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=None,
        help="Directory where the ganak binary is installed. Default: active environment bin directory.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Parallel build jobs. Default: cmake chooses.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild and overwrite an existing ganak binary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_ganak(
        repo_url=args.repo,
        commit=args.commit,
        install_dir=args.install_dir,
        jobs=args.jobs,
        force=args.force,
    )


if __name__ == "__main__":
    main()
