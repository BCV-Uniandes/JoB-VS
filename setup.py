from setuptools import setup, find_packages

# Taken and modified from https://github.com/BCV-Uniandes/ROG/blob/main/setup.py

setup(
    name='JoB-VS',
    packages=find_packages(exclude=['test']),
    package_dir={'rog': 'rog'},
    version='1.0',
    description='JoB-VS: Joint Brain-Vessel Segmentation in TOF-MRA Images',
    author='Natalia Valderrama',
    install_requires=[
        "setuptools>=18.0",
        "torch>=1.6.0",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "nibabel",
        "batchgenerators"
    ],
)