# setup.py

"""
Setup script for LiuEmbeddings package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="liuembeddings",
    version="2.0.0",
    author="Himanshu Singh",
    author_email="Himanshuclub88@gmail.com",
    description="TensorFlow-based embeddings with ChromaDB vector " \
    "store for semantic search easy embeddings and search integration. " \
    "decrease api charge as run on local light weight and easy to learn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/himanshuclub88/liuembeddings",
    packages=find_packages(),
    classifiers = [
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Text Processing :: Linguistic",
],
    python_requires=">=3.8",
    
    install_requires=[
    "sentence-transformers==5.1.2",
    "transformers==4.57.1",
    "torch==2.9.0",
    "chromadb==1.2.1",
    "numpy==1.26.4",
    ],




    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "liuembeddings=liuembeddings.cli:main",
        ],
    },
    keywords="embeddings semantic-search tensorflow chromadb nlp",
    project_urls={
        "Bug Reports": "https://github.com/himanshuclub88/liuembeddings/issues",
        "Source": "https://github.com/himanshuclub88/liuembeddings",
        "Documentation": "https://himanshuclub88.github.io/liuembeddings/",
    },
    include_package_data=True,
)
