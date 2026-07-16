"""Focused tests for the optional root GANAK adapter."""

from pathlib import Path

from wfomc.errors import ExternalToolError, GanakError
from wfomc.ganak import (
    GANAK_ARJUN_COMMIT,
    GANAK_ARJUN_REPO_URL,
    GANAK_COMMIT,
    GANAK_REPO_URL,
    ganak_count,
)


def test_ganak_count_empty_cnf_returns_one_without_binary():
    # No propositional variables means one empty model. This path must return
    # before locating GANAK because GANAK itself rejects empty input.
    assert ganak_count(0, [], {}) == 1


def test_ganak_error_belongs_to_external_tool_hierarchy():
    assert issubclass(GanakError, ExternalToolError)


def test_installer_uses_shared_ganak_metadata():
    from scripts.tools import install_ganak

    assert install_ganak.GANAK_COMMIT == GANAK_COMMIT
    assert install_ganak.GANAK_REPO_URL == GANAK_REPO_URL
    assert install_ganak.GANAK_ARJUN_COMMIT == GANAK_ARJUN_COMMIT
    assert install_ganak.GANAK_ARJUN_REPO_URL == GANAK_ARJUN_REPO_URL


def test_installer_checks_out_pinned_arjun_as_ganak_sibling(monkeypatch, tmp_path):
    from scripts.tools import install_ganak

    commands = []

    def fake_run(cmd, cwd=None, env=None):
        commands.append((cmd, cwd, env))
        if cmd[:2] == ["cmake", "--build"]:
            built = Path(cmd[2]) / install_ganak._binary_name()
            built.parent.mkdir(parents=True)
            built.touch()

    monkeypatch.setattr(install_ganak, "_run", fake_run)
    monkeypatch.setattr(install_ganak, "_require_tool", lambda _name: None)
    monkeypatch.setattr(install_ganak, "_dependency_hints", lambda: ([], {}))
    monkeypatch.setattr(
        install_ganak, "_copy_shared_libraries", lambda _build, _dir: []
    )
    monkeypatch.setattr(install_ganak, "_add_darwin_rpath", lambda _path: None)

    target = install_ganak.install_ganak(
        repo_url="https://example.test/ganak.git",
        commit="ganak-revision",
        arjun_repo_url="https://example.test/arjun.git",
        arjun_commit="arjun-revision",
        install_dir=tmp_path,
        jobs=2,
    )

    ganak_clone = commands[0]
    arjun_clone = commands[3]
    assert ganak_clone[0][:-1] == [
        "git",
        "clone",
        "--recurse-submodules",
        "https://example.test/ganak.git",
    ]
    assert Path(ganak_clone[0][-1]).name == "ganak"
    assert commands[1][0] == ["git", "checkout", "ganak-revision"]
    assert arjun_clone[0][:-1] == [
        "git",
        "clone",
        "--recurse-submodules",
        "https://example.test/arjun.git",
    ]
    assert Path(arjun_clone[0][-1]).name == "arjun"
    assert Path(arjun_clone[0][-1]).parent == Path(ganak_clone[0][-1]).parent
    assert commands[4][0] == ["git", "checkout", "arjun-revision"]
    assert target == tmp_path / install_ganak._binary_name()
    assert target.exists()


def test_copy_shared_libraries_includes_sibling_and_fetchcontent_builds(tmp_path):
    from scripts.tools import install_ganak

    build = tmp_path / "build"
    install_dir = tmp_path / "bin"
    install_dir.mkdir()
    suffix = install_ganak._shared_library_globs()[0].removeprefix("*")
    libraries = (
        build / "lib" / f"libganak{suffix}",
        build / "arjun-build" / "lib" / f"libarjun{suffix}",
        build / "_deps" / "approxmc-build" / "lib" / f"libapproxmc{suffix}",
    )
    for library in libraries:
        library.parent.mkdir(parents=True)
        library.write_bytes(library.name.encode())

    copied = install_ganak._copy_shared_libraries(build, install_dir)

    assert {path.name for path in copied} == {path.name for path in libraries}
    assert all(
        (install_dir / path.name).read_bytes() == path.name.encode()
        for path in libraries
    )
