from codecs import open
from os import path

from setuptools import setup

here = path.abspath(path.dirname(__file__))

# Read the long description from our README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "pysubtracking",
    version = "0.0.1",

    description = "Python implementation of subspace estimation and tracking methods.",
    long_description = long_description,

    # Project site
    url = "https://github.com/hasselg/pysubtracking",

    # Author details
    author = "Gregory (Greg) Hasseler",
    author_email = "ghasseler@gmail.com",

    # License info
    license = "APLv2",

    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
    ],

    keywords = "subspace estimation tracking",

    packages = ["subtracking"],

    install_requires = ["numpy", "scipy"],
)

