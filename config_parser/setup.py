from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="config_parser",
    version="1.0.0",
    author="Federico Izzo",
    author_email="federico.izzo@studenti.unitn.it",
    packages=["config_parser"],
    description="Base config parser",
    setup_requires=[],
    install_requires=requirements,
)
