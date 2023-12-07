from relik.inference.annotator import Relik
from pathlib import Path

VERSION = {}  # type: ignore
with open(Path(__file__).parent / "version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

__version__ = VERSION["VERSION"]
