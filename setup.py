from setuptools import find_packages
from setuptools_rust import RustExtension
from distutils.core import setup

NAME = 'ffp'
setup(
    name=NAME,
    version=0.1,
    author="Sebastian Pütz",
    author_email="sebastian.puetz@uni-tuebingen.de",
    license='BlueOak-1.0.0',
    packages=find_packages(),
    rust_extensions=[
        RustExtension('ffp.vocab_rs', 'Cargo.toml', debug=False)
    ],
    requires=["numpy"],
    include_package_data=True,
    zip_safe=False,
)
