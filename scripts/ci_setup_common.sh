#!/bin/bash

set -e  # Exit on any error

# Install common dependencies for Python projects
python -m pip install --upgrade pip
pip install -r mrm_requirements.txt  # Install all dependencies
pip install pylint nbconvert nbformat ipykernel jupyter

echo "Common dependencies installed."

# Function to run pylint on all Python files
run_pylint() {
  find . -type f -name "*.py" -print0 | while IFS= read -r -d '' py_code; do
      echo "pylinting $py_code..."
      pylint "$py_code"
  done
  echo "All Python files passed pylint."
}

# Function to run Jupyter notebooks in the examples folder
run_notebooks() {
  echo "Running example notebooks..."
  NOTEBOOKS=$(find examples -maxdepth 1 -print0 -type f -name "*.ipynb")
  find examples -maxdepth 1 -type f -name "*.ipynb" -print0 | while IFS= read -r -d '' nb; do
      echo "Running $nb..."
      jupyter nbconvert --to notebook --execute "$nb" --output /dev/null
  done
  echo "All notebooks ran successfully."
}

# Execute both checks
run_pylint
run_notebooks