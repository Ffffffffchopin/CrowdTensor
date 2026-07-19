import hashlib

import pytest

from scripts.community_release_build import _prepare_python_artifacts


def sha256(path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_reuse_python_artifacts_preserves_bytes_and_records_provenance(tmp_path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    wheel = source / "crowdtensord-0.2.0rc1-py3-none-any.whl"
    sdist = source / "crowdtensord-0.2.0rc1.tar.gz"
    wheel.write_bytes(b"immutable-wheel")
    sdist.write_bytes(b"immutable-sdist")
    copied_wheel, copied_sdist, provenance = _prepare_python_artifacts(
        destination,
        root=tmp_path,
        build_python="unused",
        reuse_python_artifacts_from=source,
        expected_wheel_sha256=sha256(wheel),
    )
    assert copied_wheel.read_bytes() == b"immutable-wheel"
    assert copied_sdist.read_bytes() == b"immutable-sdist"
    assert provenance["mode"] == "existing_python_artifacts_reused"
    assert provenance["wheel_rebuilt"] is False
    assert provenance["expected_wheel_sha256_enforced"] is True
    assert provenance["source_paths_public"] is False


def test_in_place_reuse_does_not_delete_python_artifacts(tmp_path) -> None:
    wheel = tmp_path / "crowdtensord-0.2.0rc1-py3-none-any.whl"
    sdist = tmp_path / "crowdtensord-0.2.0rc1.tar.gz"
    wheel.write_bytes(b"wheel")
    sdist.write_bytes(b"sdist")
    before = (wheel.read_bytes(), sdist.read_bytes())
    selected_wheel, selected_sdist, provenance = _prepare_python_artifacts(
        tmp_path,
        root=tmp_path,
        build_python="unused",
        reuse_python_artifacts_from=tmp_path,
        expected_wheel_sha256=sha256(wheel),
    )
    assert (selected_wheel.read_bytes(), selected_sdist.read_bytes()) == before
    assert provenance["sdist_rebuilt"] is False


def test_reuse_rejects_wrong_or_missing_expected_wheel_hash(tmp_path) -> None:
    (tmp_path / "crowdtensord-0.2.0rc1-py3-none-any.whl").write_bytes(b"wheel")
    (tmp_path / "crowdtensord-0.2.0rc1.tar.gz").write_bytes(b"sdist")
    for expected in ("", "sha256:" + "0" * 64):
        with pytest.raises(ValueError):
            _prepare_python_artifacts(
                tmp_path,
                root=tmp_path,
                build_python="unused",
                reuse_python_artifacts_from=tmp_path,
                expected_wheel_sha256=expected,
            )
