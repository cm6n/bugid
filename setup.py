from setuptools import setup, find_packages

setup(
    name="cli-tool",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Add your dependencies here
    ],
    entry_points={
        'console_scripts': [
            'cli-tool=cli_tool.cli:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A command line tool template",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/cli-tool",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
