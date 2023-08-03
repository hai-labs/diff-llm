import pathlib

from setuptools import find_packages, setup

__version__ = "0.0.0a0"


with open("requirements.txt") as f:
    install_requires = [x.strip() for x in f.readlines()]

LICENSE: str = "Apache"
README: str = pathlib.Path("README.md").read_text()


setup(
    name="diff-llm",
    version=__version__,
    author="Niels Bantilan",
    author_email="niels.bantilan@gmail.com",
    description="LLM that predicts text diffs.",
    long_description=README,
    long_description_content_type="text/markdown",
    license=LICENSE,
    keywords=["machine-learning", "artificial-intelligence"],
    data_files=[("", ["LICENSE"])],
    include_package_data=True,
    packages=find_packages(
        include=["diff_llm*"],
    ),
    python_requires=">3.7",
    platforms="any",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        f"License :: OSI Approved :: {LICENSE} Software License",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
    url="https://github.com/hai-labs/diff-llm",
    project_urls={
        "Source Code": "https://github.com/hai-labs/diff-llm",
        "Issue Tracker": "https://github.com/hai-labs/diff-llm/issues",
    },
)
