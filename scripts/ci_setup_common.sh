#!/bin/bash

# Install common dependencies for Python projects
python -m pip install --upgrade pip
pip install -r mrm_requirements.txt  # Install all dependencies
pip install pylint nbconvert nbformat ipykernel jupyter

echo "Common dependencies installed."

# Function to run pylint on all Python files
run_pylint() {
  find pymrm examples exercises -type f -name "*.py" -print0 | while IFS= read -r -d '' py_code; do
      echo "pylinting $py_code..."
      pylint --score=y --exit-zero --disable=all --enable=syntax-error \
             --msg-template="{path}: Overall Score: {score}" "$py_code"
  done
  echo "All Python files passed pylint."
}

# Function to run Jupyter notebooks in the examples folder
run_notebooks() {
  set -e  # Exit on any error

  # Install the local package and nbclient for executing notebooks
  python -m pip install -e .
  python -m pip install nbclient ipykernel

  echo "Running example notebooks..."

  # Run each notebook using nbclient
  find examples -maxdepth 1 -type f -name "*.ipynb" -print0 | while IFS= read -r -d '' nb; do
      echo "Running $nb..."
      python -c "
import nbformat
from nbclient import NotebookClient

with open('$nb') as f:
    nb = nbformat.read(f, as_version=4)

client = NotebookClient(nb)
client.execute()
"
  done

  echo "All notebooks ran successfully."
}

# Execute both checks
run_pylint
run_notebooks