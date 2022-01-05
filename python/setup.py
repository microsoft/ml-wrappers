# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Setup file for ml-wrappers package."""
import os
import shutil

from setuptools import find_packages, setup

with open('ml_wrappers/version.py') as f:
    code = compile(f.read(), f.name, 'exec')
    exec(code)

README_FILE = 'README.md'
LICENSE_FILE = 'LICENSE.txt'

# Note: used when generating the wheel but not on pip install of the package
if os.path.exists('../LICENSE'):
    shutil.copyfile('../LICENSE', LICENSE_FILE)


CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Operating System :: Microsoft :: Windows',
    'Operating System :: MacOS',
    'Operating System :: POSIX :: Linux'
]

DEPENDENCIES = [
    'numpy',
    'pandas',
    'scipy',
    'scikit-learn'
]

with open(README_FILE, 'r', encoding='utf-8') as f:
    README = f.read()

setup(
    name=name,  # noqa: F821
    version=version,  # noqa: F821
    description='Machine Learning Wrappers SDK for Python',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Microsoft Corp',
    author_email='ilmat@microsoft.com',
    license='MIT License',
    url='https://github.com/microsoft/ml-wrappers',
    classifiers=CLASSIFIERS,
    packages=find_packages(exclude=["*.tests"]),
    install_requires=DEPENDENCIES,
    zip_safe=False
)
