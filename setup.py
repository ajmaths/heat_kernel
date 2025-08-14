from setuptools import setup, find_packages

setup(
    name="heat_kernel_package",
    version="0.1.0",
    author="Washington Mio, Wenwen Li",
    author_email="wli11uco@gmail.com",
    description="Heat kernel and diffusion distance computations on graphs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ajmaths/heat_kernel.git",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "networkx",
        "scipy",
        "pygsp"
    ],
    python_requires=">=3.7",
)
