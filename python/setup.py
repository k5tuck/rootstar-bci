"""Setup script for Neural Fingerprint package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent.parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="neural-fingerprint",
    version="0.1.0",
    author="Rootstar BCI Team",
    author_email="",
    description="Neural Fingerprint Detection & Sensory Simulation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/k5tuck/rootstar-bci",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "sqlalchemy>=2.0.0",
        "websockets>=11.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black",
            "isort",
            "mypy",
        ],
        "faiss": [
            "faiss-cpu>=1.7.4",
        ],
        "faiss-gpu": [
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "nf-collect=neural_fingerprint.collection:main",
            "nf-vr-bridge=neural_fingerprint.vr_bridge:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
