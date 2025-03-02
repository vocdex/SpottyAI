from setuptools import setup, find_packages

setup(
    name="spotty",  # Replace with your package name
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
        "License :: OSI Approved :: MIT License",  # Replace with your license
    ],
    python_requires=">=3.8",  # Specify your minimum Python version
    entry_points={
        "console_scripts": [
            "spotty-cli=spotty.main_interface:main",  # Replace with your entry-point
        ]
    },
)

