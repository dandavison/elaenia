from setuptools import find_packages
from setuptools import setup


setup(
    name="elaenia",
    author="Dan Davison",
    author_email="dandavison7@gmail.com",
    description="Identify bird vocalizations",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
