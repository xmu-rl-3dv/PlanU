#!/usr/bin/env python

"""Package BlockWorld reasoners with the repository's shared PlanU core.

Benchmark datasets, prompts, PDDL files, and planner binaries remain external
experiment inputs under ``blockworld/examples`` and are not package data.
"""

import os
from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

packages = find_packages(
    include=["planu_core", "planu_core.*"],
) + find_packages(
    where="blockworld",
    include=["reasoners", "reasoners.*"],
)

setup(
    name="llm-reasoners",
    version="1.0.2",
    description=(
        "A library for advanced reasoning methods with large language models"
    ),
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=packages,
    package_dir={"reasoners": "blockworld/reasoners"},
    entry_points={
        "console_scripts": [
            "reasoners-visualizer=reasoners.visualization:main",
        ],
    },
    install_requires=[
        "tqdm",
        "fire",
        "numpy",
        "scipy",
        "torch",
        "datasets",
        "huggingface_hub",
        "transformers",
        "sentencepiece",
        "openai",
        "tarski",
        "peft",
        "optimum",
        "ninja",
        "bitsandbytes",
        "fairscale",
        "google-generativeai",
        "anthropic",
    ],
    include_package_data=True,
    python_requires=">=3.9",
)
