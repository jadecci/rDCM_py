from setuptools import setup, find_packages

setup(
    name='rdcmpy',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.7, <4',
    install_requires=[
        'numpy>=1.24',
        'scipy>=1.10',
    ],
    extras_require={
        'dev': [
            'flake8',
            'pyre-check',
            'pytest',
            'pytest-cov',
            'pandas',
        ],
    },
    entry_points={
        'console_scripts': [
            'rdcm=rdcmpy.main:main',
            ]
    },
)
