#!/usr/bin/env python3
"""
Tests for scripts/generate_model_card.py using the real template in templates/model_card.md.

These tests verify:
- Conditional processing for quantized_models
- Variable replacement for common fields
- File writing via save_model_card
"""

import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_scripts_to_path() -> None:
    scripts_dir = _repo_root() / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))


def test_model_card_generator_conditionals_truthy(tmp_path):
    _add_scripts_to_path()
    from generate_model_card import ModelCardGenerator

    template_path = _repo_root() / "templates" / "model_card.md"
    generator = ModelCardGenerator(str(template_path))

    variables = {
        "model_name": "My Fine-tuned Model",
        "model_description": "A test description.",
        "repo_name": "user/repo",
        "base_model": "HuggingFaceTB/SmolLM3-3B",
        "dataset_name": "OpenHermes-FR",
        "training_config_type": "Custom",
        "trainer_type": "SFTTrainer",
        "batch_size": "8",
        "gradient_accumulation_steps": "16",
        "learning_rate": "5e-6",
        "max_epochs": "3",
        "max_seq_length": "2048",
        "hardware_info": "CPU",
        "experiment_name": "exp-123",
        "trackio_url": "https://trackio.space/exp",
        "dataset_repo": "tonic/trackio-experiments",
        "author_name": "Unit Tester",
        "quantized_models": True,
    }

    content = generator.generate_model_card(variables)

    # Conditional: when True, the quantized tag should appear
    assert "- quantized" in content

    # Common variables replaced in multiple locations
    assert "base_model: HuggingFaceTB/SmolLM3-3B" in content
    assert "trainer_type: SFTTrainer" in content
    assert 'from_pretrained("user/repo")' in content
    assert "Hardware\": \"CPU\"" not in content  # ensure no escaped quotes left
    assert "hardware: \"CPU\"" in content

    # Save to file and verify
    output_path = tmp_path / "README_test.md"
    assert generator.save_model_card(content, str(output_path)) is True
    assert output_path.exists()
    assert output_path.read_text(encoding="utf-8") == content


def test_model_card_generator_conditionals_falsey(tmp_path):
    _add_scripts_to_path()
    from generate_model_card import ModelCardGenerator

    template_path = _repo_root() / "templates" / "model_card.md"
    generator = ModelCardGenerator(str(template_path))

    variables = {
        "model_name": "My Model",
        "model_description": "A test description.",
        "repo_name": "user/repo",
        "base_model": "HuggingFaceTB/SmolLM3-3B",
        "dataset_name": "OpenHermes-FR",
        "training_config_type": "Custom",
        "trainer_type": "SFTTrainer",
        "batch_size": "8",
        "learning_rate": "5e-6",
        "max_epochs": "3",
        "max_seq_length": "2048",
        "hardware_info": "CPU",
        "quantized_models": False,
    }

    content = generator.generate_model_card(variables)

    # Conditional: quantized tag should be absent
    assert "- quantized" not in content

    # The if/else block is removed by current implementation when False
    assert "{{#if quantized_models}}" not in content
    assert "{{/if}}" not in content

    # Variable replacement still occurs elsewhere
    assert "base_model: HuggingFaceTB/SmolLM3-3B" in content
    assert 'from_pretrained("user/repo")' in content

    # Save to file
    output_path = tmp_path / "README_no_quant.md"
    assert generator.save_model_card(content, str(output_path)) is True
    assert output_path.exists()


def test_model_card_generator_variable_replacement(tmp_path):
    _add_scripts_to_path()
    from generate_model_card import ModelCardGenerator

    template_path = _repo_root() / "templates" / "model_card.md"
    generator = ModelCardGenerator(str(template_path))

    base_model = "custom/base-model"
    repo_name = "custom/repo-name"
    variables = {
        "model_name": "Var Test Model",
        "model_description": "Testing variable replacement.",
        "repo_name": repo_name,
        "base_model": base_model,
        "dataset_name": "dataset-x",
        "trainer_type": "SFTTrainer",
        "batch_size": "4",
        "gradient_accumulation_steps": "1",
        "max_seq_length": "1024",
        "hardware_info": "CPU",
        "quantized_models": False,
    }

    content = generator.generate_model_card(variables)

    assert f"base_model: {base_model}" in content
    assert f'from_pretrained("{repo_name}")' in content
    assert "trainer_type: SFTTrainer" in content


