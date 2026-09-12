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

OPTIONAL_DEPENDENCIES = {
    "anthropic": ["anthropic"],
    "exllama": ["huggingface_hub", "ninja"],
    "gemini": ["google-generativeai"],
    "llama": ["fairscale"],
    "llama-cpp": ["llama-cpp-python", "scipy"],
    "openai": ["openai", "optimum"],
    "quantization": ["bitsandbytes"],
    "rnd": ["DI-engine", "easydict"],
}
OPTIONAL_DEPENDENCIES["all"] = sorted(
    {
        dependency
        for requirements in OPTIONAL_DEPENDENCIES.values()
        for dependency in requirements
    }
)

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
    author="Ziwei Deng",
    author_email="dengziwei@stu.xmu.edu.cn",
    url="https://github.com/xmu-rl-3dv/PlanU",
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
        "accelerate",
        "datasets",
        "numpy",
        "pddl==0.2.0",
        "peft",
        "PyYAML",
        "requests",
        "sentencepiece",
        "tarski",
        "torch",
        "tqdm",
        "transformers",
    ],
    extras_require=OPTIONAL_DEPENDENCIES,
    include_package_data=True,
    python_requires=">=3.9",
)
