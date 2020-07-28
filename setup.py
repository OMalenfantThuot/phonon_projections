from setuptools import setup

requirements = ["numpy>=1.18", "abipy>=0.8.0"]

setup(
    name="phonon_projections",
    description="Package for phonon projections for Raman project.",
    packages=["phonon_projections"],
    install_requires=requirements,
    classifiers=["Programming Language :: Python :: 3.7"],
)
