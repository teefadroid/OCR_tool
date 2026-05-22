"""
Test configuration for PharmOCR.

The ingestion / layout / output stages depend on heavy native libraries
(numpy, opencv, pdf2image, PyMuPDF). For unit testing of the deterministic
post-processing logic (numerals, BiDi, INN matching, audit log), we stub
those imports so that the tests run in any clean Python environment.

If you want to run the full integration tests (touching the real OCR
models), install requirements.txt first.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# Make the project importable when running `pytest` from the repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _stub(module_name: str, attrs: dict | None = None) -> None:
    """Register a lightweight stub module if the real one is unavailable."""
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
        return
    except Exception:
        pass
    mod = types.ModuleType(module_name)
    for k, v in (attrs or {}).items():
        setattr(mod, k, v)
    sys.modules[module_name] = mod


# Stub numpy with the bare minimum surface used by the imports we need
class _FakeNdarray:
    pass


_stub("numpy", {"ndarray": _FakeNdarray})

# Stub cv2, pdf2image, PIL — only used in ingestion which we do not exercise here
_stub("cv2")
_stub("pdf2image", {"convert_from_path": lambda *a, **k: []})
_stub("PIL")
_stub("PIL.Image")
