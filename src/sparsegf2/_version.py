"""Single source of truth for the package version.

``sparsegf2.__version__`` is imported from here; ``pyproject.toml``
declares ``dynamic = ["version"]`` and reads this same attribute via
``[tool.setuptools.dynamic]``. This file is the only place the version
string lives; do not duplicate it into ``pyproject.toml``.
"""

__version__ = "2.1.0"
