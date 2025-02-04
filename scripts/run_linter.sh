#!/bin/bash

# The commands provided here will be executed for the linting step in the ci
# but may also be used locally
# execute from root folder!

flake8 --output-file=test/linter_errors.txt ./src/
flake8_junit test/linter_errors.txt test/linter_errors.xml
