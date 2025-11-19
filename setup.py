from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="prompt-entropy-experiment",
    version="0.1.0",
    author="Ibrahim Cesar",
    author_email="ibrahim@ibrahimcesar.com",
    description="Information-theoretic analysis of prompt engineering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ibrahimcesar/prompt-entropy-experiment",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "statsmodels>=0.14.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "sentence-transformers>=2.2.0",
        "nltk>=3.8.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "jupyter>=1.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
)
