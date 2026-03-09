from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

extensions = [
    Extension(
        "wfomc.utils.multinomial",
        ["src/wfomc/utils/multinomial.pyx"],
    ),
    Extension(
        "wfomc.cell_graph.cell_graph",
        ["src/wfomc/cell_graph/cell_graph.pyx"],
    ),
    Extension(
        "wfomc.algo.IncrementalWFOMC",
        ["src/wfomc/algo/IncrementalWFOMC.pyx"],
    ),
    Extension(
        "wfomc.algo.RecursiveWFOMC",
        ["src/wfomc/algo/RecursiveWFOMC.pyx"],
    ),
    Extension(
        "wfomc.algo.FastWFOMC",
        ["src/wfomc/algo/FastWFOMC.pyx"],
    ),
    Extension(
        "wfomc.algo.StandardWFOMC",
        ["src/wfomc/algo/StandardWFOMC.pyx"],
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        language_level=3,
        annotate=True,          # generates .html annotation files
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "nonecheck": False,
            "cdivision": True,
        },
    )
)
