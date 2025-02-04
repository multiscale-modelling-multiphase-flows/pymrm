import nbformat
import argparse
from nbclient import NotebookClient


def run(nb):
    with open(nb) as f:
        nb = nbformat.read(f, as_version=4)
        client = NotebookClient(nb)
        client.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a dedicated jupyter notebook from a python script")
    parser.add_argument("-f", "--file", type=str, help="Path to the input notebook (.ipynb)")

    args = parser.parse_args()

    nb = args.file

    run(nb)
