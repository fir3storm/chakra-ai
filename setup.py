from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="chakra-ai",
    version="0.1.0",
    description="Chakra-AI / KimiPy Engine: Windows 8GB RAM Lossless Streaming MoE Engine for Kimi K3 by Abhirup Guha (Info Security Solution)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Abhirup Guha (Info Security Solution)",
    author_email="abhirupguha@infosecuritysolution.com",
    url="https://github.com/InfoSecuritySolution/KimiWin-Py",
    packages=find_packages(include=["kimipy", "kimipy.*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "psutil>=5.8.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
