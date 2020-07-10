from os import path

import setuptools

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README_pypi.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

extra_requirements = {}
with open("requirements_optional.txt") as f:
    extra_reqs = f.read().splitlines()
for req in extra_reqs:
    pkg, feature = req.split("#")
    extra_requirements[feature.strip()] = [pkg.strip()]

setuptools.setup(
    name="pygpso",
    version="0.5",
    description="Bayesian optimisation method leveraging Gaussian Processes "
    "surrogate",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jajcayn/pygpso",
    author="Nikola Jajcay",
    author_email="nikola.jajcay@gmail.com",
    license="MIT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    extras_require=extra_requirements,
    include_package_data=True,
)
