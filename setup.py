import os

from setuptools import find_packages
from setuptools import setup


setup(
    name="elaenia",
    author="Dan Davison",
    author_email="dandavison7@gmail.com",
    description="Segment recordings of bird vocalizations",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    # install_requires=["jupyter", "librosa", "matplotlib", "numpy"],
)
