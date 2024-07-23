# setup.py

from setuptools import setup, find_packages

setup(
    name="CourtCheck",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision",
        "opencv-python",
        "matplotlib",
        "numpy",
        "git+https://github.com/facebookresearch/detectron2.git",
        # Add other dependencies here
    ],
)
