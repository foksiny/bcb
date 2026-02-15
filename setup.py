from setuptools import setup, find_packages

setup(
    name="bcb",
    version="1.0.6",
    author="Antigravity",
    description="A minimalist compiler for BCB (Basic Compiler Backend) language",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'bcb = bcb.main:main',
        ],
    },
    python_requires='>=3.10',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
