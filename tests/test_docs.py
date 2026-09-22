from pathlib import Path


def test_readme_references_supported_workflows():
    readme = Path("README.md").read_text(encoding="utf-8")

    for text in ("train.py", "notebooks/train.ipynb", "play_gui.py", "RTX 4070", "epoch_000001.pt", "CUDA"):
        assert text in readme


def test_project_metadata_references_runtime_packages():
    metadata = Path("pyproject.toml").read_text(encoding="utf-8")
    requirements = Path("requirements.txt").read_text(encoding="utf-8")

    assert "python-shogi" in metadata and "PySide6" in metadata
    assert "torch" in requirements and "PyYAML" in requirements
