# Project Mobile Robotics

This repository provides the codebase and an initial structure for the Mobile Robotics project of the Chair of 
Automatic Control at FAU. The intention of the repository is to provide you with all the required tools and examples that allow a 
quick head start into the project.

## Getting started

We assume that you have some experience with the command line, python and git.

If not have a look at the following tutorials, 
- Command line using Ubuntu: https://ubuntu.com/tutorials/command-line-for-beginners#1-overview
- Command line using Windows: https://www.cs.princeton.edu/courses/archive/spr05/cos126/cmd-prompt.html
- a full documentation for **Git** is provided by https://git-scm.com/book/en/v2.
- The W3School provide a very good tutorial for python https://www.w3schools.com/python/

We use the Spot Python SDKs. Therefore, we highly recommend to have a look at https://dev.bostondynamics.com/readme for 
the conceptual documentation, the installation process and the Python client library.

### required software

- GIT: https://git-scm.com/download/win
- Python 3.8.10 (Please use this version!):  https://www.python.org/downloads/release/python-3810/
- Pycharm Community: https://www.jetbrains.com/pycharm/download/?section=windows

### Project Repository

create a workspace directory on your computer. Clone the repository in it and update the submodules

    git clone https://gitlab.cs.fau.de/lrt/practical-seminar-mobile-robotics.git
    cd  practical-seminar-mobile-robotics
    git submodule update --init --recursive

### Setting up a virtual python environment (venv)

We recommend using Pycharm for this process.

- open a project. Select practical-seminar-mobile-robotics as the root folder
- Define the interpreter as a venv (select Python 3.8.10 here as a base interpreter)
- install required packages Normally it installs them automatically. 

You can change the interpreter later on if you go to `File->Settings: Project <Project-Name> -> Python Interpreter` you can see all installed packages in the venv. 

If you have selected the correct python interpreter you should be able to see the 'bosdyn' packages.
Additionally, the Terminal in Pycharm should indicate that you are using the venv by showing the name of the venv in brackets.

Note that in Windows the virtual environment is only active in the `Command Promt`! 

#### activating venv in the command line
If correctly setup, pycharm automatically activates the right Python environment in the command prompt. If not use 

activate venv: 
    
    source ~/.../venv/bin/activate

deactivate venv: 
    
    deactivate

for Windows

    .\venv\Scripts\activate

If you failed to install the requirements, run the following command in the root directory of this repo (ensure that you activated the venv):

`pip install -r requirements.txt` 

This installs all the required packages. During the project you might need additional packages to perform your tasks. 
To update the requirements.txt file, run the following command in the root directory of your repo:

`pip freeze --local > requirements.txt`.

The local flag ensures that only the requirements of the venv are used. Do so after you have installed new packages and 
want to commit your code.

To ensure that your team is using the latest package requirements run 

`pip install -r requirements.txt` 

after you have pulled your code 

## Hello Spot

... the first in-person session of this course will cover the basics like communicating with Spot and using the robot_wrapper as a basis for your project.

## Important information 

### WIFI:
    SSID: spot-BD-10400003
    password: q3ezxuygl9l0 // - l not 1!

### Connecting/ Authentification
    Username: student
    Password: LRT-Spot03025

### robot IP address:
    192.168.80.3

