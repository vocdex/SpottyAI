from setuptools import find_packages, setup

setup(
    name="spotty",
    version="0.1.0",
    author="Shukrullo Nazirjonov",
    author_email="nazirjonovsh2000@gmail.com",
    description="Conversation agent for Spot legged robot",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    # install_requires=open("requirements.txt").read().splitlines(),  # Add dependencies here
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.9",  # Specify your minimum Python version
    entry_points={
        "console_scripts": [
            "spotty-cli=spotty.main_interface:main",
        ]
    },
)
