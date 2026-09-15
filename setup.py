from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "reservoir._sumtree",
            sources=[
                "src/reservoir/csrc/sumtreemodule.c",
                "src/reservoir/csrc/sumtree.c",
            ],
            extra_compile_args=["-O3"],
            # -march=native intentionally omitted: unsafe for distributed packages.
            # Users wanting native tuning can set CFLAGS="-O3 -march=native" before install.
        )
    ]
)
