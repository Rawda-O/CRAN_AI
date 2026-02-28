from setuptools import setup, find_packages

# Minimal setup.py for editable installs (pip install -e .)
setup(
    name="cran-ai",
    version="0.1.0",
    description="6G-ready AI-native Cognitive Relay Framework",
    packages=find_packages(exclude=("tests", "notebooks", "runs", "docs")),
    python_requires=">=3.11,<3.12",
)
