import os.path

import setuptools

with open("requirements.txt") as fh:
    requirements = [line for line in fh.readlines() if not line.startswith("--")]

setuptools.setup(
    name="smallnn",
    version=1.0,
    author="Francisco Eiras",
    author_email="francisco.girbal@gmail.com",
    python_requires=">=3.0",
    description="Small NN package with networks implemented from scratch",
    url="https://github.com/fgirbal/aims-ml-nn",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "Operating System :: POSIX :: Linux",
    ],
    install_requires=requirements,
    include_package_data=True,
)
