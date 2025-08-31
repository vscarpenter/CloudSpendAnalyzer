"""Setup configuration for AWS Cost Explorer CLI."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aws-cost-explorer-cli",
    version="0.1.0",
    author="Vinny Carpenter",
    author_email="vinny@vinny.dev",
    description="A CLI tool for querying AWS cost data using natural language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinnycarpenter/aws-cost-explorer-cli",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "boto3>=1.26.0",
        "click>=8.0.0",
        "rich>=12.0.0",
        "pyyaml>=6.0",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aws-cost-cli=aws_cost_cli.cli:main",
        ],
    },
)
