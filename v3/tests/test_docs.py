"""The Python examples in README.md run, in order, as written."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _blocks(path: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", path.read_text(), flags=re.S)


@pytest.mark.parametrize("doc", ["README.md"])
def test_readme_examples_run(doc):
    blocks = _blocks(ROOT / doc)
    assert blocks, f"no python examples found in {doc}"
    namespace: dict = {}
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, f"{doc} block {i + 1}", "exec"), namespace)
        except Exception as exc:  # pragma: no cover - the message is the point
            raise AssertionError(f"{doc}, python block {i + 1} failed:\n{code}") from exc
