import os

from setuptools import find_packages
from setuptools import setup
from distutils.extension import Extension

need_cython = not all([os.path.exists(p) for p in
                       ["ffp/subwords/hash_indexers.c", "ffp/subwords/ngrams.c", "ffp/subwords/explicit_indexer.c"]])

if need_cython:
    from Cython.Build import cythonize
    extensions = cythonize([Extension("ffp.subwords.hash_indexers", ["ffp/subwords/hash_indexers.pyx"]),
                            Extension("ffp.subwords.ngrams", ["ffp/subwords/ngrams.pyx"]),
                            Extension("ffp.subwords.explicit_indexer", ["ffp/subwords/explicit_indexer.pyx"])])
else:
    extensions = [Extension("ffp.subwords.hash_indexers", ["ffp/subwords/hash_indexers.c"]),
                  Extension("ffp.subwords.ngrams", ["ffp/subwords/ngrams.c"])]

NAME = 'ffp'
setup(
    name=NAME,
    version="0.1.2",
    author="Sebastian Pütz",
    author_email="seb.puetz@gmail.com",
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Text Processing :: Linguistic",
    ),
    description="Interface to finalfusion embeddings",
    ext_modules=extensions,
    license='BlueOak-1.0.0',
    packages=find_packages(),
    install_requires=["numpy", "toml"],
    include_package_data=True,
    package_data={'': ['*.pyx']},
    url="https://github.com/sebpuetz/ffp"
)
