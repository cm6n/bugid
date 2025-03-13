from setuptools import setup, find_packages

setup(
    name="bugid",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "librosa>=0.10.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "soundfile>=0.12.0",
        "tensorflow==2.18.1",
    ],
    entry_points={
        "console_scripts": [
            "bugid=cli_tool.cli:cli"
        ]
    },
    python_requires=">=3.8",
    description="A library for identifying bugs in audio recordings using machine learning",
    author="Chris",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
