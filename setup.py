from setuptools import setup, find_packages

setup(
    name="synthetic_benchmarking",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/my_package",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pytest",
        "pytz",
        "python-dotenv",
        "PyGithub",
        "gitpython",
        "tabulate",
        "posthog",
        "ipython",
        "ipdb",
        "pytest",
        "tiktoken",
        "pydantic",
    ],
    python_requires=">=3.6",
)
