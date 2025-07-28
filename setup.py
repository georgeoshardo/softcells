#!/usr/bin/env python3
"""
Setup script for SoftCells - Soft Body Physics Simulation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="softcells",
    version="1.0.0",
    author="SoftCells Development Team",
    author_email="dev@softcells.example.com",
    description="A comprehensive soft body physics simulation system",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/softcells",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Games/Entertainment :: Simulation",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "softcells=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 