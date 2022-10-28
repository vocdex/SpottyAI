# Project Mobile Robotics

This repository provides the codebase and an initial structure for the Mobile Robotics project of the Chair of 
Automatic Control at FAU.
The intention of the repo is to provide you with all the required tools and examples that allow a 
quick head start into the project.

## Getting started

We assume that you have some experience with the command line, python and git.

If not, 
- Command line using Ubuntu: https://ubuntu.com/tutorials/command-line-for-beginners#1-overview
- Command line using Windows: https://www.cs.princeton.edu/courses/archive/spr05/cos126/cmd-prompt.html
- a full documentation for **Git** is provided by https://git-scm.com/book/en/v2.
- The W3School provide a very good tutorial for python https://www.w3schools.com/python/

We use the Spot Python SDKs. Therefore, we highly recommend to have a look at https://dev.bostondynamics.com/readme for 
the conceptual documentation, the installation process and the Python client library.

The Spot SDK Repo has a lot of examples. It is added as a submodule. 
To load the submodules run the following commands in the command line of the root directory of the repo:

``` 
git submodule init 
git submodule update
```

### Install Python and setting up a virtual environment (venv)

We follow the same process as described in BostonDynamics Quickstart https://dev.bostondynamics.com/docs/python/quickstart#system-setup.

Proceed until *'Install Spot Python packages'*. All proceeding steps are either covered by this **README** or will be demonstrated during the first session of the project

### Package requirements

Our code requires external packages. The most important ones are the spot python packages to communicate with the robot.
You could manually install all the required packages, but this might be inefficient.  
To ensure that you and your team is always using the same packages, a `requirements.txt` file can be very useful.

To keep our package management clean we only want to make the packages of the venv a requirement.
Ensure that you have activated the virtual environment (as proposed by BostonDynamics)
indicated by the name of the venv in brackets in the command line.

I have already created a requirements.txt file which includes the most basic packages to get you started. 

To install them, run the following command in the root directory of this repo:

`pip install -r requirements.txt` 

This installs all the required packages. During the project you might need additional packages to perform your tasks. 
To update the requirements.txt file, run the following command in the root directory of your repo:

`pip freeze --local > requirements.txt`.

The local flag ensures that only the requirements of the venv are used. Do so after you have installed new packages and 
want to commit your code.

To ensure that your team is using the latest package requirements run 

`pip install -r requirements.txt` 

after you have pulled your code 

### Integrated Development Environment (IDE) 

Before we start to develop code for our robot, we will set up the IDE for an easier development.
We recommend PyCharm. Feel free to use any other IDE, but we cannot provide any support for these.

#### Setup PyCharm (Community)
after installing PyCharm Community (https://www.jetbrains.com/pycharm/download) we need to set up a new project.

Define the location of the project (use this repo)
select a python interpreter (use the previously configured virtual environment e.g. `…/spot_venv/bin/python3`)

You can change the interpreter at any time or have a look at the installed packages.

If you go to `File->Settings: Project <Project-Name> -> Python Interpreter` you can see all installed packages in the venv. 

If you have selected the correct python interpreter you should be able to see the 'bosdyn' packages.
Additionally, the Terminal in Pycharm should indicate that you are using the venv by showing the name of the venv in brackets.

Note that in Windows the virtual environment is only active in the `Command Promt`! 

## Hello Spot

... the first session of the project will cover the basics like communicating with Spot.

#### remember to launch your code in a venv
activate venv: `source ~/.../spot_env/bin/activate`

deactivate venv: `deactivate`

#### WIFI:
SSID: `spot-BD-10400003`
password: `q3ezxuygl9l0` - l not 1!

#### Connecting
Username: `user` 
Password: `c037gcf6n93f`

for convenience during the examples:

    export BOSDYN_CLIENT_USERNAME=user
    export BOSDYN_CLIENT_PASSWORD=c037gcf6n93f 

for windows

    set BOSDYN_CLIENT_USERNAME=user
    set BOSDYN_CLIENT_PASSWORD=c037gcf6n93f 

#### robot IP address:
`192.168.80.3`

