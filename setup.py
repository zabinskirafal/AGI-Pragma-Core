# Minimal setup.py for backwards compatibility with tools that
# don't yet support PEP 517/518 (e.g. older pip, editable installs).
# All authoritative configuration is in pyproject.toml.
from setuptools import setup

if __name__ == "__main__":
    setup()
