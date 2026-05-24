"""Tests for the host-fingerprint capture helpers."""

from __future__ import annotations

from unittest import mock

from agns.utils.host_info import (
    BLAS_ENV_VARS,
    capture_host_info,
    fmt_host_fingerprint,
)

# ---------------------------------------------------------------------------
# capture_host_info
# ---------------------------------------------------------------------------


def test_capture_returns_expected_keys() -> None:
    info = capture_host_info()
    expected = {
        "cpu_model",
        "platform",
        "python_version",
        "numpy_version",
        "scipy_version",
        "blas_threads_env",
    }
    assert expected <= set(info.keys())


def test_capture_blas_env_covers_every_documented_var() -> None:
    info = capture_host_info()
    assert set(info["blas_threads_env"].keys()) == set(BLAS_ENV_VARS)


def test_capture_versions_are_non_empty_strings() -> None:
    info = capture_host_info()
    for key in ("python_version", "numpy_version", "scipy_version"):
        assert isinstance(info[key], str)
        assert info[key] != ""


def test_capture_is_idempotent_on_same_machine() -> None:
    """Two calls in the same process return identical fingerprints.

    Time-of-call fields are deliberately not captured; everything in
    the dict should be a static property of the machine + environment.
    """
    a = capture_host_info()
    b = capture_host_info()
    assert a == b


def test_capture_picks_up_env_overrides(monkeypatch) -> None:
    """A test-set env var is reflected in the snapshot."""
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    info = capture_host_info()
    assert info["blas_threads_env"]["OMP_NUM_THREADS"] == "7"


# ---------------------------------------------------------------------------
# fmt_host_fingerprint
# ---------------------------------------------------------------------------


def test_fmt_empty_input_returns_empty_string() -> None:
    assert fmt_host_fingerprint(None) == ""
    assert fmt_host_fingerprint({}) == ""


def test_fmt_uniform_blas_threads_summarised_compactly() -> None:
    fp = fmt_host_fingerprint(
        {
            "cpu_model": "Test CPU",
            "blas_threads_env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        }
    )
    assert "Test CPU" in fp
    assert "BLAS threads = 1" in fp


def test_fmt_heterogeneous_blas_threads_spelled_out() -> None:
    fp = fmt_host_fingerprint(
        {
            "cpu_model": "Test CPU",
            "blas_threads_env": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "4"},
        }
    )
    # When env vars disagree, the formatter lists each one explicitly so
    # a reader can tell which library is mis-pinned.
    assert "OMP_NUM_THREADS=1" in fp
    assert "MKL_NUM_THREADS=4" in fp


def test_fmt_empty_blas_env_says_unspecified() -> None:
    fp = fmt_host_fingerprint(
        {
            "cpu_model": "Test CPU",
            "blas_threads_env": {"OMP_NUM_THREADS": "", "MKL_NUM_THREADS": ""},
        }
    )
    assert "unspecified" in fp


def test_fmt_missing_cpu_falls_back_to_unknown() -> None:
    fp = fmt_host_fingerprint({"cpu_model": "", "blas_threads_env": {}})
    assert "unknown CPU" in fp


def test_cpu_model_falls_back_to_platform_processor_when_no_proc_cpuinfo() -> None:
    """When /proc/cpuinfo is unavailable (non-Linux), fall back to platform.processor.

    Mock both code paths so the test runs identically on every OS.
    """
    with (
        mock.patch("agns.utils.host_info.Path") as fake_path,
        mock.patch("agns.utils.host_info.platform.processor", return_value="Mock CPU"),
    ):
        fake_path.return_value.is_file.return_value = False
        info = capture_host_info()
    assert info["cpu_model"] == "Mock CPU"
