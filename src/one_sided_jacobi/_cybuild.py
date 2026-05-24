"""Cython build configuration — defines _core extension for one_sided_jacobi.

Referenced by pyproject.toml via [tool.setuptools.cmdclass].
Overrides the ``build`` command to inject Cython extensions before sub-commands run.
"""
from setuptools import Extension
from setuptools.command.build import build as _build
from Cython.Build import cythonize


class build(_build):
    """Custom build that injects Cython extensions before sub-commands run."""

    def finalize_options(self) -> None:
        extensions = [
            Extension(
                "one_sided_jacobi._core",
                ["src/one_sided_jacobi/_core.pyx"],
            ),
        ]
        self.distribution.ext_modules = cythonize(
            extensions,
            language_level=3,
        )
        super().finalize_options()
