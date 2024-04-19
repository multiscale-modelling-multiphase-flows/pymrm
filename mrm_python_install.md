In the Multiphase Reactor Modeling class we will use Python. For computations we will use the standard packages, Numpy and Scipy, and our own package: pymrm.

## Option 1: using Anaconda
A suggested setup under Windows is using Anaconda (for Python) in combination with VS Code.

#### Step 1: Install Anaconda
1. Download the [Anaconda installer](https://www.anaconda.com/download) for your operating system from the Anaconda website.
2. Follow the installation instructions on the website to install Anaconda.

#### Step 2: Set Up the Virtual Environment
1. Open the Anaconda Prompt (or your terminal on macOS/Linux).
2. Navigate to the directory containing your mrm_environment.yml file.
3. Create the Anaconda environment by running:
`conda env create -f mrm_environment.yml` 
This command creates a new environment named `mrm` configured with the necessary Python packages.

#### Step 3: Activate the Environment
Activate the newly created environment by running: `conda activate mrm`
#### Step 4: Install Visual Studio Code (VS Code)

1. Download and install VS Code from the official website.
2. Install the Python extension for VS Code by Microsoft from the VS Code marketplace.

#### Step 5: Configure VS Code
1. Open VS Code.
2. Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P)` and type: `Python: Select Interpreter`
3. Choose the Python interpreter from your mrm environment.

--------------
# Python Environment Setup for MRM Class

This guide provides step-by-step instructions for students on setting up their Python development environment using the built-in `venv` module and Visual Studio Code. We will also cover the installation of required packages using a `requirements.txt` file.

## Step 1: Install Python

1. **Download Python:** Visit the [official Python website](https://www.python.org/downloads/) and download the latest version suitable for your operating system.
2. **Install Python:** Run the downloaded installer. Ensure to select "Add Python to PATH" before clicking "Install Now".

## Step 2: Create and Activate a Virtual Environment

A virtual environment is a self-contained directory that contains a Python installation for a particular version of Python, plus a number of additional packages.

1. Open your command prompt (Windows) or terminal (macOS or Linux).
2. Navigate to your project directory where you want the virtual environment to be located.
3. Run the following command to create a virtual environment named `mrm`:
   ```bash
   python -m venv mrm

--------------




## Option 2: Using Python venv and requirements.txt
If you prefer not to use Anaconda, you can set up your environment using Python's built-in venv module.

#### Step 1: Install Python
1. Download Python from the [official Python website](https://www.python.org/downloads/).
2. Follow the instructions to install Python on your system. 
#### Step 2: Create and Activate a Virtual Environment
1. Open your terminal or command prompt.
2. Navigate to your project directory or where you want to place your virtual environment.
3. Run the following command to create a virtual environment named mrm: `python -m venv mrm`
Activate the virtual environment:
On Windows: `mrm\Scripts\activate`
On macOS/Linux: `source mrm/bin/activate`
#### Step 3: Install Required Packages
1. Ensure you are in the directory containing the mrm_requirements.txt file.
2. Install the required packages using: `pip install -r mrm_requirements.txt`

#### Step 4: Install and Configure VS Code
1. Follow the same installation and configuration instructions for VS Code as described in Option 1, steps 4 and 5.