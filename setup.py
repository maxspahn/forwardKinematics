import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="forwardKinematics",
    version="0.0.1",
    author="maxspahn",
    author_email="m.spahn@tudelft.nl",
    description="Forward kinematics for casadi.",
    long_description=long_description,
    url="https://github.com/maxspahn/forwardKinematics",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy", "casadi", "matplotlib",],
    extras_require={"urdf": ['urdf2casadi']}, 
    python_requires=">=3.6",
)
