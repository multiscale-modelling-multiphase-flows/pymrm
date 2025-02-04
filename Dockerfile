# Filename Dockerfile
# This Dockerfile can be used to create an image which can be used as an environment for pymrm.
# The final image features:
# - OS: debian bookworm
# - libraries: python-is-python3
# - default entrypoint: /home/user
#
# To create the image, call:
#     docker build -t <name of image>:<version> .
# Once you're satisfied with the image, give it a tag, e.g.
#     docker tag <name of image> <name of repo>/<name of image>:<version>
# And finally push:
#     docker push <name of repo>/<name of image>:<version>

# Set debian as base layer
FROM python:3.10-bookworm

# add flake8
RUN apt-get update
RUN apt-get -y install flake8

# add new user
RUN useradd -m -s /bin/bash user
USER user
WORKDIR /home/user

# add virtual environment with preinstalled libraries
# PYTHONUNBUFFERED=1 enforces immediate printing to terminal
ENV PYTHONUNBUFFERED=1
RUN python -m venv .venv
ENV PATH="/home/user/.venv/bin:$PATH"
RUN yes | python -m pip install --upgrade pip
RUN yes | python -m pip install numpy scipy numba pandas matplotlib ipython ipykernel pytest pytest-cov sphinx sphinx_rtd_theme nbclient flake8-junit-report
