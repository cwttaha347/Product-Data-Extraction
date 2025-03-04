#!/usr/bin/env python
"""
Setup script for the Product Data Extraction Platform.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(
    name="product-data-extractor",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A platform for extracting product data from PDF documents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/product-data-extractor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "pdf-extractor=run_pipeline:main",
        ],
    },
)
