"""Shared helpers for the per-module circuit notebooks.

Every ``_build_<module>.py`` imports :func:`md`, :func:`code`, and
:func:`build_and_execute` from here, so each build script holds only its
cell *content*. The notebooks are executed against the project root so
``import sparsegf2`` resolves to the editable install.

Run any builder from the project root::

    .venv/bin/python notebooks/circuits/_build_<module>.py
"""

from __future__ import annotations

import pathlib

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent


def md(text: str):
    """A markdown cell."""
    return new_markdown_cell(text)


def code(text: str):
    """A code cell."""
    return new_code_cell(text)


def build_and_execute(out_name: str, cells: list, *, timeout: int = 180) -> None:
    """Assemble ``cells`` into a notebook, execute it, and write it out.

    Parameters
    ----------
    out_name : str
        File name written under ``notebooks/circuits/`` (e.g. ``"graphs.ipynb"``).
    cells : list
        The ordered list of cells from :func:`md` / :func:`code`.
    timeout : int
        Per-cell execution timeout in seconds.
    """
    nb = new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    out = HERE / out_name
    nbformat.write(nb, out)
    print(f"wrote + executed {out.relative_to(PROJECT_ROOT)}")
