from pathlib import Path
import shutil
import tempfile
import pytest


def test_default_config():
    from unstructured_inference.config import inference_config

    assert inference_config.TT_TABLE_CONF == 0.5


def test_env_override(monkeypatch):
    monkeypatch.setenv("TT_TABLE_CONF", 1)
    from unstructured_inference.config import inference_config

    assert inference_config.TT_TABLE_CONF == 1


@pytest.fixture()
def _setup_tmpdir():
    from unstructured_inference.config import inference_config

    _tmpdir = tempfile.tempdir
    _storage_tmpdir = inference_config.INFERENCE_GLOBAL_WORKING_PROCESS_DIR
    _storage_tmpdir_bak = f"{inference_config.INFERENCE_GLOBAL_WORKING_PROCESS_DIR}_bak"
    if Path(_storage_tmpdir).is_dir():
        shutil.move(_storage_tmpdir, _storage_tmpdir_bak)
        tempfile.tempdir = None
    yield
    if Path(_storage_tmpdir_bak).is_dir():
        if Path(_storage_tmpdir).is_dir():
            shutil.rmtree(_storage_tmpdir)
        shutil.move(_storage_tmpdir_bak, _storage_tmpdir)
        tempfile.tempdir = _tmpdir


@pytest.mark.usefixtures("_setup_tmpdir")
def test_env_storage_disabled(monkeypatch):
    monkeypatch.setenv("INFERENCE_GLOBAL_WORKING_DIR_ENABLED", "false")
    from unstructured_inference.config import inference_config

    assert not inference_config.INFERENCE_GLOBAL_WORKING_DIR_ENABLED
    assert str(Path.home() / ".cache/unstructured") == inference_config.INFERENCE_GLOBAL_WORKING_DIR
    assert not Path(inference_config.INFERENCE_GLOBAL_WORKING_PROCESS_DIR).is_dir()
    assert tempfile.gettempdir() != inference_config.INFERENCE_GLOBAL_WORKING_PROCESS_DIR


@pytest.mark.usefixtures("_setup_tmpdir")
def test_env_storage_enabled(monkeypatch):
    monkeypatch.setenv("INFERENCE_GLOBAL_WORKING_DIR_ENABLED", "true")
    from unstructured_inference.config import inference_config

    assert inference_config.INFERENCE_GLOBAL_WORKING_DIR_ENABLED
    assert str(Path.home() / ".cache/unstructured") == inference_config.INFERENCE_GLOBAL_WORKING_DIR
    assert Path(inference_config.INFERENCE_GLOBAL_WORKING_PROCESS_DIR).is_dir()
    assert tempfile.gettempdir() == inference_config.INFERENCE_GLOBAL_WORKING_PROCESS_DIR
