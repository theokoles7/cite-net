"""Cite-Net setup utility."""

from setuptools import find_packages, setup

setup(
    name =              "cite-net",
    version =           "0.0.1",
    author =            "Gabriel C. Trahan",
    author_email =      "gabriel.trahan1@louisiana.edu",
    description =       (
                        "This repository showcases my work on a Data Mining project peratining to "
                        "node classification."
                        ),
    license =           "MIT",
    url =               "https://github.com/theokoles7/cite-net",
    packages =          find_packages(),
    python_requires =   ">=3.10",
    install_requires =  [
        "matplotlib",
        "numpy",
        "pandas",
        "PyWavelets",
        "scikit-learn",
        "termcolor",
        "torch",
        "torchvision",
        "tqdm"
    ]
)