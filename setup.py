from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="chakra-ai",
    version="0.2.0",
    description="Chakra AI - Agentic Coding Terminal with Multi-Engine Kimi K3 Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Abhirup Guha (Info Security Solution)",
    url="https://github.com/fir3storm/chakra-ai",
    packages=find_packages(include=["chakra", "chakra.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
    ],
    entry_points={
        "console_scripts": [
            "chakra=chakra.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
