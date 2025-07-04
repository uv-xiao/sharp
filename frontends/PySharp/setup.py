# PySharp setup.py

from setuptools import setup, find_packages

setup(
    name="pysharp",
    version="0.1.0",
    description="Pythonic hardware description for Sharp",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        # These would be provided by the Sharp build
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Compilers",
        "Topic :: System :: Hardware",
        "Programming Language :: Python :: 3",
    ],
)