def test_default_config():
    from unstructured_inference.config import inference_config

    assert inference_config.TABLE_IMAGE_CROP_PAD == 12


def test_env_override(monkeypatch):
    monkeypatch.setenv("TABLE_IMAGE_CROP_PAD", 1)
    from unstructured_inference.config import inference_config

    assert inference_config.TABLE_IMAGE_CROP_PAD == 1
