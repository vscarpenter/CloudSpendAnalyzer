"""Setup configuration for simplified AWS Cost CLI."""

from setuptools import setup, find_packages

setup(
    name="aws-cost-cli-simple",
    version="2.0.0",
    description="Simple AWS Cost Explorer CLI - Query costs using natural language",
    author="AWS Cost CLI Team",
    python_requires=">=3.8",
    py_modules=[
        "cli",
        "config", 
        "models",
        "llm",
        "aws",
        "cache",
        "query",
        "utils"
    ],
    install_requires=[
        "boto3>=1.26.0",
        "click>=8.0.0",
        "requests>=2.28.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
    },
    entry_points={
        "console_scripts": [
            "aws-cost-cli=cli:main",
        ],
    },
)
