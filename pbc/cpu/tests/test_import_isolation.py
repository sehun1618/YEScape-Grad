"""Stage-0 package and legacy-import isolation tests."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

import pyscf_pbc_dcdft


FORBIDDEN = {"pyscf_pbc_dc", "pyscf_pbc_dc_restricted"}
SOURCE_ROOT = Path(__file__).parents[1] / "src" / "pyscf_pbc_dcdft"


class ImportIsolationTest(unittest.TestCase):
    def test_public_imports(self) -> None:
        self.assertEqual(pyscf_pbc_dcdft.__version__, "0.1.0a0")
        self.assertTrue(hasattr(pyscf_pbc_dcdft.dft, "DCRKS"))
        self.assertTrue(hasattr(pyscf_pbc_dcdft.dft, "DCKRKS"))

    def test_source_has_no_legacy_runtime_imports(self) -> None:
        offenders: list[str] = []
        for path in SOURCE_ROOT.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(name.split(".")[0] in FORBIDDEN for name in names):
                    offenders.append(str(path.relative_to(SOURCE_ROOT)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
