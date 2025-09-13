#!/usr/bin/env python3
"""
Tests for scripts/push_to_huggingface.py focusing on model card creation/upload.

We mock Hugging Face Hub interactions and create dummy model folders to verify:
- Repo id resolution via whoami
- Repository creation call
- README.md upload with expected content (fallback simple card path)
- Uploading of model files from the directory
"""

import sys
import types
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_scripts_to_path() -> None:
    scripts_dir = _repo_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def _make_full_model_dir(base: Path) -> Path:
    model_dir = base / "full_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    # Create an empty weight file to satisfy validation
    (model_dir / "model.safetensors").write_bytes(b"")
    return model_dir


def _make_lora_model_dir(base: Path) -> Path:
    model_dir = base / "lora_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "adapter_model.bin").write_bytes(b"\x00")
    return model_dir


def test_push_model_card_full_model(monkeypatch, tmp_path):
    _add_scripts_to_path()
    import push_to_huggingface as mod

    # Ensure module thinks HF is available and patch API + functions
    monkeypatch.setattr(mod, "HF_AVAILABLE", True, raising=False)

    create_repo_calls = []
    upload_file_calls = []

    class DummyHfApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"name": "testuser"}

    def fake_create_repo(*, repo_id, token=None, private=False, exist_ok=False, repo_type=None):
        create_repo_calls.append({
            "repo_id": repo_id,
            "token": token,
            "private": private,
            "exist_ok": exist_ok,
            "repo_type": repo_type,
        })

    def fake_upload_file(*, path_or_fileobj, path_in_repo, repo_id, token, repo_type=None):
        path = Path(path_or_fileobj)
        content = None
        if path.exists() and path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                content = None
        upload_file_calls.append({
            "path_in_repo": path_in_repo,
            "repo_id": repo_id,
            "token": token,
            "repo_type": repo_type,
            "content": content,
            "local_path": str(path),
        })

    monkeypatch.setattr(mod, "HfApi", DummyHfApi, raising=False)
    monkeypatch.setattr(mod, "create_repo", fake_create_repo, raising=False)
    monkeypatch.setattr(mod, "upload_file", fake_upload_file, raising=False)

    # Prepare dummy full model directory
    model_dir = _make_full_model_dir(tmp_path)

    pusher = mod.HuggingFacePusher(
        model_path=str(model_dir),
        repo_name="my-repo",
        token="fake-token",
        private=True,
        author_name="Tester",
        model_description="Desc",
        model_name="BaseModel",
        dataset_name="DatasetX",
    )

    # Execute push (this should use fallback simple model card)
    ok = pusher.push_model(
        training_config={"param": 1},
        results={"train_loss": 0.1, "eval_loss": 0.2, "perplexity": 9.9},
    )
    assert ok is True

    # Repo creation was called with resolved user prefix
    assert any(c["repo_id"] == "testuser/my-repo" for c in create_repo_calls)

    # README upload occurred and contains either generator or fallback content (full model)
    readme_calls = [c for c in upload_file_calls if c["path_in_repo"] == "README.md"]
    assert readme_calls, "README.md was not uploaded"
    readme_content = readme_calls[-1]["content"] or ""
    assert (
        "fine-tuned Voxtral ASR model" in readme_content
        or "SmolLM3" in readme_content
        or "Model Details" in readme_content
    )
    assert "DatasetX" in readme_content or "Training Configuration" in readme_content

    # Model files were uploaded (config and weights)
    uploaded_paths = {c["path_in_repo"] for c in upload_file_calls}
    assert "config.json" in uploaded_paths
    assert "model.safetensors" in uploaded_paths


def test_push_model_card_lora_model_fallback(monkeypatch, tmp_path):
    _add_scripts_to_path()
    import push_to_huggingface as mod

    # Ensure module thinks HF is available and patch API + functions
    monkeypatch.setattr(mod, "HF_AVAILABLE", True, raising=False)

    upload_file_calls = []

    class DummyHfApi:
        def __init__(self, token=None):
            self.token = token

        def whoami(self):
            return {"username": "anotheruser"}

    def fake_create_repo(*, repo_id, token=None, private=False, exist_ok=False, repo_type=None):
        return None

    def fake_upload_file(*, path_or_fileobj, path_in_repo, repo_id, token, repo_type=None):
        path = Path(path_or_fileobj)
        content = None
        if path.exists() and path.is_file():
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                content = None
        upload_file_calls.append({
            "path_in_repo": path_in_repo,
            "repo_id": repo_id,
            "content": content,
        })

    monkeypatch.setattr(mod, "HfApi", DummyHfApi, raising=False)
    monkeypatch.setattr(mod, "create_repo", fake_create_repo, raising=False)
    monkeypatch.setattr(mod, "upload_file", fake_upload_file, raising=False)

    # Insert a dummy generate_model_card module that raises in generate to force fallback
    dummy_mod = types.ModuleType("generate_model_card")

    class RaisingGen:
        def __init__(self, *args, **kwargs):
            pass

        def generate_model_card(self, variables):
            raise RuntimeError("force fallback")

    def default_vars():
        return {}

    dummy_mod.ModelCardGenerator = RaisingGen
    dummy_mod.create_default_variables = default_vars
    sys.modules["generate_model_card"] = dummy_mod

    # Prepare dummy lora model directory
    model_dir = _make_lora_model_dir(tmp_path)

    pusher = mod.HuggingFacePusher(
        model_path=str(model_dir),
        repo_name="my-lora-repo",
        token="fake-token",
        private=False,
        author_name="Tester",
        model_description="Desc",
        model_name="BaseModel",
        dataset_name="DatasetY",
    )

    ok = pusher.push_model(training_config={}, results={})
    assert ok is True

    # README upload occurred and contains either generator or fallback content (LoRA)
    readme_calls = [c for c in upload_file_calls if c["path_in_repo"] == "README.md"]
    assert readme_calls, "README.md was not uploaded"
    readme_content = readme_calls[-1]["content"] or ""
    assert (
        "LoRA adapter for Voxtral ASR" in readme_content
        or "SmolLM3" in readme_content
        or "Model Details" in readme_content
    )
    assert "DatasetY" in readme_content or "Training Configuration" in readme_content

    # LoRA files uploaded
    uploaded_paths = {Path(c.get("local_path", "")).name for c in upload_file_calls if c.get("local_path")}
    assert any(name.startswith("adapter_") for name in uploaded_paths)


