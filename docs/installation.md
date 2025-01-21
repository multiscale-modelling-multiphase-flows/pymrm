# Installation Guide

In the Multiphase Reactor Modeling class, we will use Python for computations. We will utilize standard packages such as NumPy and SciPy, along with our own package: PyMRM.

## Step 1: Installing Python

1. **Download and Install Python**:
   - Go to the [official Python website](https://www.python.org/downloads/).
   - Download the latest version of Python for your operating system.
   - Follow the installation instructions for your operating system.
   - Ensure that you check the option to add Python to your system PATH during installation.

2. **Verify Python Installation**:
   - Open a terminal or command prompt.
   - Run the following command to verify the installation:
     ```sh
     python --version
     ```
   - You should see the installed Python version.

## Step 2: Setting Up a Virtual Environment

1. **Create a Virtual Environment**:
   - Open a terminal or command prompt.
   - Navigate to your project directory.
   - Run the following command to create a virtual environment:
     ```sh
     python -m venv venv
     ```

2. **Activate the Virtual Environment**:
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```
   - On macOS and Linux:
     ```sh
     source venv/bin/activate
     ```

3. **Verify Virtual Environment Activation**:
   - Run the following command to verify that the virtual environment is activated:
     ```sh
     which python
     ```
   - The output should point to the Python executable within the virtual environment.

## Step 3: Installing Required Packages

1. **Upgrade pip**:
   - Run the following command to upgrade pip to the latest version:
     ```sh
     pip install --upgrade pip
     ```

2. **Install NumPy and SciPy**:
   - Run the following command to install NumPy and SciPy:
     ```sh
     pip install numpy scipy
     ```

3. **Install Matplotlib**:
   - Run the following command to install Matplotlib for plotting:
     ```sh
     pip install matplotlib
     ```

4. **Install PyMRM**:
   - If PyMRM is available on PyPI, run the following command:
     ```sh
     pip install pymrm
     ```
   - If you are installing from a local version, navigate to the directory containing `setup.py` and run:
     ```sh
     pip install .
     ```

## Step 4: Verifying the Installation

1. **Create a Test Script**:
   - Create a new Python file named `test_installation.py`.
   - Add the following code to the file:
     ```python
     import numpy as np
     import scipy
     import matplotlib.pyplot as plt
     import pymrm

     print("NumPy version:", np.__version__)
     print("SciPy version:", scipy.__version__)
     print("Matplotlib version:", plt.__version__)
     print("PyMRM version:", pymrm.__version__)
     ```

2. **Run the Test Script**:
   - Run the following command to execute the test script:
     ```sh
     python test_installation.py
     ```
   - Verify that the versions of the installed packages are printed without any errors.

## Conclusion

You have successfully installed Python, set up a virtual environment, and installed the required packages for the Multiphase Reactor Modeling class. You are now ready to start using PyMRM for your computations.
