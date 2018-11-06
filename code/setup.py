import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pydif",
    version="0.0.1",
    author="Hamish Nicholson",
    author_email="hamish_nicholson@college.harvard.edu",
    description="A work in progress package to implement automatic differentiation for numeric functions in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pydif/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)