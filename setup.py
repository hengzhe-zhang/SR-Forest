#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
]

setup_requirements = ['pytest-runner']

test_requirements = ['pytest>=3']

setup(
    author="Hengzhe Zhang",
    author_email='hengzhe.zhang@ecs.vuw.ac.nz',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    description="An open source python library for GP-based ensemble learning methods",
    install_requires=requirements,
    license="BSD license",
    long_description=readme,
    include_package_data=True,
    keywords='sr_forest',
    name='sr_forest',
    packages=find_packages(include=['sr_forest', 'sr_forest.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/zhenlingcn/sr_forest',
    version='0.1.0',
    zip_safe=False,
)
