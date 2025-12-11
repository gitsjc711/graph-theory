# setup.py
import sys
from cx_Freeze import setup, Executable

build_exe_options = {
    "packages": ["tkinter", "matplotlib", "numpy", "networkx"],
    "include_files": [],
    "excludes": []
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="GraphAlgorithmVisualizer",
    version="1.0",
    description="图算法可视化工具",
    options={"build_exe": build_exe_options},
    executables=[Executable("graph_algorithm_app.py", base=base)]
)