from setuptools import setup, find_packages

setup(
    name='Runner',
    version='0.0.1',
    description='A sample Python package',
    author='John Doe',
    author_email='jdoe@example.com',
    packages=find_packages(
        where='src'
    ),
    install_requires=[
        'numpy',
        'pandas',
    ],
)