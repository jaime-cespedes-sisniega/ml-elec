from setuptools import setup

with open('requirements/setup_requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='ml-pipeline',
    version='1.0.0',
    packages=['ml_pipeline'],
    install_requires=required
)
