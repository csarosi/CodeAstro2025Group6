import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="emcee rapper", # Replace with your own username
    version="0.0.1",
    author="Cat, Thomas, Anthony, Andres",
    description="emcee wrapper",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/csarosi/emcee_rapper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.10',
)
