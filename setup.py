import setuptools
from setuptools import setup

install_deps = ["ipykernel", "matplotlib", "scikit-learn", "statsmodels"]
dev_deps = ["black", "isort", "nbstripout", "pylint"]

with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="hrl",
    license="MIT",
    author="Will Long",
    author_email="wlong@princeton.edu",
    description="Using HMMs and RL to understand lapses in decision-making.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wlong799/hrl",
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=install_deps,
    extras_require={"dev": dev_deps},
)
