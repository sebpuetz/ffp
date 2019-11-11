from setuptools import find_packages
from distutils.core import setup

NAME = 'ffp'
test_requires = ['pytest']
setup(
    name=NAME,
    version=0.1,
    author="Sebastian Pütz",
    author_email="sebastian.puetz@uni-tuebingen.de",
    license='BlueOak-1.0.0',
    packages=find_packages(),
    test_requires=test_requires,
    include_package_data=True,
    zip_safe=False,
)
