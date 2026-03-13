from pathlib import Path

_pkg_dir = Path(__file__).resolve().parent
_src_pkg = _pkg_dir.parent / "src" / "sim"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))

