#!/usr/bin/env python3

"""FunASR setup script."""

import os

from setuptools import find_packages
from setuptools import setup


requirements = {
    "install": [
        "scipy>=1.4.1",
        "librosa",
        "jamo",  # For kss
        "PyYAML>=5.1.2",
        "soundfile>=0.12.1",
        "kaldiio>=2.17.0",
        "torch_complex",
        # "nltk>=3.4.5",
        "sentencepiece",  # train
        "jieba",
        "rotary_embedding_torch",
        # "ffmpeg-python",
        # "pypinyin>=0.44.0",
        # "espnet_tts_frontend",
        # ENH
        "pytorch_wpe",
        "editdistance>=0.5.2",
        # "g2p",
        # "nara_wpe",
        # PAI
        "oss2",
        # "edit-distance",
        # "textgrid",
        # "protobuf",
        "tqdm",
        "umap_learn",
        "jaconv",
        "hydra-core>=1.3.2",
        "tensorboardX",
        # "rotary_embedding_torch",
        "requests",
    ],
    # train: The modules invoked when training only.
    "train": [
        "editdistance",
    ],
    # all: The modules should be optionally installled due to some reason.
    #      Please consider moving them to "install" occasionally
    "all": [
        # NOTE(kamo): Append modules requiring specific pytorch version or torch>1.3.0
        "torch_optimizer",
        "fairscale",
        "transformers",
        "openai-whisper",
    ],
    "setup": [
        "numpy",
        "pytest-runner",
    ],
    "test": [
        "pytest>=3.3.0",
        "pytest-timeouts>=1.2.1",
        "pytest-pythonpath>=0.7.3",
        "pytest-cov>=2.7.1",
        "hacking>=2.0.0",
        "mock>=2.0.0",
        "pycodestyle",
        "jsondiff<2.0.0,>=1.2.0",
        "flake8>=3.7.8",
        "flake8-docstrings>=1.3.1",
        "black",
    ],
    "doc": [
        "Jinja2",
        "Sphinx",
        "sphinx-rtd-theme>=0.2.4",
        "sphinx-argparse>=0.2.5",
        "commonmark",
        "recommonmark>=0.4.0",
        "nbsphinx>=0.4.2",
        "sphinx-markdown-tables>=0.0.12",
        "configargparse>=1.2.1",
    ],
    "llm": [
        "transformers>=4.32.0",
        "accelerate",
        "tiktoken",
        "einops",
        "transformers_stream_generator>=0.0.4",
        "scipy",
        "torchvision",
        "pillow",
        "matplotlib",
    ],
}
requirements["all"].extend(requirements["train"])
requirements["all"].extend(requirements["llm"])
requirements["test"].extend(requirements["train"])

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {k: v for k, v in requirements.items() if k not in ["install", "setup"]}

dirname = os.path.dirname(__file__)
version_file = os.path.join(dirname, "funasr", "version.txt")
with open(version_file, "r") as f:
    version = f.read().strip()
setup(
    name="funasr",
    version=version,
    url="https://github.com/alibaba-damo-academy/FunASR.git",
    author="Speech Lab of Alibaba Group",
    author_email="funasr@list.alibaba-inc.com",
    description="FunASR: A Fundamental End-to-End Speech Recognition Toolkit",
    long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="The MIT License",
    packages=find_packages(include=["funasr*"]),
    package_data={"funasr": ["version.txt"]},
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    python_requires=">=3.7.0",
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "funasr = funasr.bin.inference:main_hydra",
            "funasr-train = funasr.bin.train:main_hydra",
            "funasr-export = funasr.bin.export:main_hydra",
            "scp2jsonl = funasr.datasets.audio_datasets.scp2jsonl:main_hydra",
            "jsonl2scp = funasr.datasets.audio_datasets.jsonl2scp:main_hydra",
            "funasr-scp2jsonl = funasr.datasets.audio_datasets.scp2jsonl:main_hydra",
            "funasr-jsonl2scp = funasr.datasets.audio_datasets.jsonl2scp:main_hydra",
        ]
    },
)
