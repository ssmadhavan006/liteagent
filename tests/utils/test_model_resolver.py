import pytest
from liteagent.utils.model_resolver import get_ollama_paths, resolve_model_path

def test_get_ollama_paths(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODELS", "/custom/ollama/models")
    paths = get_ollama_paths()
    assert paths[0] == "/custom/ollama/models"
    assert len(paths) > 1

def test_resolve_model_path_unknown():
    with pytest.raises(ValueError) as exc_info:
        resolve_model_path("non_existent_model_tag_123")
    assert "Unknown model tag" in str(exc_info.value)
