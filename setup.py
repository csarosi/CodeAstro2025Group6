from setuptools import setup, find_packages

setup(
    name="emcee_rapper",
    version="0.0.1",
    description="A wrapper around emcee with plotting tools",
    author="Cat, Thomas, Anthony, Andres",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "corner",
        "matplotlib",
        "emcee",
    ],
    python_requires=">=3.10",
)
